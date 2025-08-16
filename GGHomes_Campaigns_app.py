# app.py
import gzip
import streamlit as st
import pandas as pd
import base64
from io import StringIO, BytesIO
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide", page_title="GGHomes Campaign Planner")

# ---------- CONSTANTS ----------
# CHANGED: one vendor only
mail_vendor_ids = {
    "GG": 1
}

# CHANGED: consolidated costs under a single "GG" vendor
costs = {
    "GG": {
        "Anderson- Tan Kraft": 0.55,
        "Anderson- Pressure Sealed": 0.55,
        "Anderson- Pressure Sealed Green": 0.55,
        "Anderson- Pressure Sealed 2": 0.55,
        "Anderson- Streetview Postcard": 0.55,
        "Anderson- Check Letter": 0.55,
        "Anderson- Check Letter 2": 0.55,
        "Anderson- Unbranded Check Letter": 0.55,
        "YLHQ- Pressure Sealed": 0.6,
        "YLHQ- Letters W/ Window Envelope": 0.6,
        "YLHQ- Streetview Postcard": 0.54,
        "YLHQ- Streetviw Valuation": 0.54,
        "YLHQ- Testimonial Card": 0.579,
        "YLHQ- Thank You Card": 1.13,
        "Redstone- Pressure Sealed": 0.47,
        "Redstone- Check Letter": 0.47,
        "Redstone- Unbranded Pressure Sealed": 0.47,
        "Redstone- Unbranded Check Letter": 0.47,
        "Redstone- We Missed You": 0.5,
        "Redstone- CC Referral Mailer": 0.75,
        "Redstone- CC Offer Mailer": 0.75,
        "Redstone- We Missed You Offer": 0.45,
        "Redstone- Testimonial": 0.5,
        "Redstone- Streetview Postcard": 0.48,
        "Redstone- Streetview Valuation": 0.5,
        "Redstone- Thank You Card": 0.5,
        "Redstone- 3 Tier Check": 0.5,
        "Redstone- Check Letter 2": 0.5,
        "Redstone- Tan Kraft": 0.5,
        "Redstone- We Missed You Unbranded": 0.45
    }
}

# ---------- HELPERS ----------
@st.cache_data
def calculate_pieces(budget):
    cost_per_piece = 1
    return budget / cost_per_piece

@st.cache_data
def max_possible_budget(num_rows):
    cost_per_piece = 1
    return num_rows * cost_per_piece

def update_costs_for_first_class():
    additional_cost = 0.12
    if 'first_class_costs_updated' not in st.session_state:
        for vendor in costs:
            costs[vendor] = {k: v + additional_cost for k, v in costs[vendor].items()}
        st.session_state['first_class_costs_updated'] = True

def update_file_budget(file_name, new_budget):
    st.session_state['file_budgets'][file_name] = new_budget

def calculate_total_budget():
    return sum(st.session_state['file_budgets'].values())

def update_total_budget_display():
    total_budget = sum(st.session_state['file_budgets'].values())
    return total_budget

@st.cache_data
def create_downloadable_file(data: pd.DataFrame, base_name: str):
    """
    Returns an auto-click download link for a CSV.GZ
    """
    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="w") as gz_file:
        data.to_csv(gz_file, index=False)
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    href = (
        f'<a href="data:application/gzip;base64,{b64}" '
        f'download="{base_name}.csv.gz" id="download-link">Download file</a>'
    )
    script = """
    <script>
    var link = document.getElementById('download-link');
    if (link) { link.click(); }
    </script>
    """
    return href + script

@st.cache_data
def distribute_and_download(df, weekly_distribution, input_type, file_length, file_name, total_pieces, i):
    combined_data_list = []

    # Normalize columns
    df.columns = df.columns.str.lower()

    # Ensure county
    if 'county' not in df.columns:
        df['county'] = 'Unknown'
    else:
        df['county'] = df['county'].fillna('Unknown')

    # Shuffle inside county groups
    grouped = df.groupby('county').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)

    num_rows_per_group_dict = {}

    for split_number, value in weekly_distribution.items():
        if input_type == "Percentage":
            num_rows = int(total_pieces * value / 100)
        else:
            num_rows = value

        num_rows_per_group = (
            np.floor(grouped['county'].value_counts(normalize=True) * num_rows)
            .round()
            .astype(int)
        )
        num_rows_per_group_dict[split_number] = num_rows_per_group

        split_data_list = []
        for name, group in grouped.groupby('county'):
            to_take = min(num_rows_per_group[name], len(group))
            split_data_list.append(group.iloc[:to_take])
            num_rows -= to_take

        while num_rows > 0:
            for name, group in grouped.groupby('county'):
                if num_rows == 0:
                    break
                if len(group) > num_rows_per_group_dict[split_number][name]:
                    to_take = min(num_rows, len(group) - num_rows_per_group_dict[split_number][name])
                    split_data_list.append(
                        group.iloc[
                            num_rows_per_group_dict[split_number][name]:
                            num_rows_per_group_dict[split_number][name] + to_take
                        ]
                    )
                    num_rows -= to_take

        split_data = pd.concat(split_data_list)
        grouped = grouped.drop(split_data.index)

        split_data['File Name'] = file_name

        for j in range(st.session_state[f'drops_{split_number}']):
            drop_date = st.session_state[f'drop_date_{split_number}_{j}']
            letter_type = st.session_state[f'letter_type_{split_number}_{j+1}']
            tracking_number = st.session_state[f'tracking_number_{split_number}_{j+1}']
            repeated = split_data.copy()
            repeated['Drop Date'] = drop_date
            repeated['Letter Type'] = letter_type
            repeated['Tracking Number'] = tracking_number
            combined_data_list.append(repeated)

    combined_data = pd.concat(combined_data_list)
    return combined_data

# ---------- MAIN APP LOGIC ----------
def run_gghomes_campaigns_app(cleaned_files=None):
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = {}
    if 'file_budgets' not in st.session_state:
        st.session_state['file_budgets'] = {}

    if not cleaned_files:
        st.error("Please upload a file in the 'file uploader' (left sidebar) before proceeding.")
        return

    st.title('Campaign Planner')
    left, right = st.columns([2, 1])

    with left:
        # CHANGED: single option "GG"
        mail_vendor = st.selectbox(
            "Select Mailer",
            ["GG"],
            key="mail_vendor",
            index=0
        )

    with right:
        first_class = st.checkbox("Apply First Class (+$0.12 per piece)", key="first_class_toggle")
        if first_class:
            update_costs_for_first_class()

    # Sidebar: file selection + processed list
    file_names = list(cleaned_files.keys())
    selected_file_name = st.sidebar.selectbox("Select a file", ["No File Selected"] + file_names)
    st.sidebar.markdown("<br>" * 3, unsafe_allow_html=True)

    processed_files = list(st.session_state['processed_data'].keys())
    if processed_files:
        st.sidebar.markdown("### Processed files:")
        for fn in processed_files:
            st.sidebar.write(fn)
    else:
        st.sidebar.markdown("No files have been processed yet.")

    # Per-file processing
    if selected_file_name != "No File Selected":
        selected_file_data = cleaned_files[selected_file_name]

        # Reset per-file state
        st.session_state['weekly_distribution'] = {}
        st.session_state['adjusted_budget'] = 0
        total_input = 0

        st.markdown(f"### Processing file: {selected_file_name}")

        cleaned_file = pd.read_csv(StringIO(selected_file_data['data'].decode('utf-8')))
        cleaned_file.name = selected_file_name
        df = cleaned_file
        df.columns = df.columns.str.lower()
        file_length = len(df)
        max_budget = max_possible_budget(file_length)
        budget = max_budget
        total_pieces = calculate_pieces(budget)

        if file_length < total_pieces:
            st.warning(f"File doesn't have enough rows. The maximum budget for this file is: ${int(max_budget)}")
            st.session_state['adjusted_budget'] = int(max_budget)
            budget = int(max_budget)
            total_pieces = calculate_pieces(budget)
            st.rerun()
        else:
            file_processed = True

        input_type = "Percentage"
        num_splits = st.number_input(
            f'Enter the number of splits for {selected_file_name}',
            min_value=1, max_value=52, step=1, key=f"num_splits_{selected_file_name}"
        )

        st.session_state['selected_file_data'] = df

        if file_processed:
            for i in range(1, num_splits + 1):
                cols = st.columns(4)
                key = f"{selected_file_name}_split_{i}"
                drop_date_key = f"drop_date_{i}"
                tracking_number_key = f"tracking_number_{i}"
                drops_key = f"drops_{i}"

                if drop_date_key not in st.session_state:
                    st.session_state[drop_date_key] = pd.to_datetime('today')
                if tracking_number_key not in st.session_state:
                    st.session_state[tracking_number_key] = ''

                with cols[0]:
                    if input_type == "Percentage":
                        initial_value = 100 / num_splits
                        value = st.number_input(
                            f"Split {i} (%)",
                            min_value=0.0, max_value=100.0,
                            value=float(initial_value), key=key
                        )
                        drops = st.number_input("Drops", min_value=1, format="%d", key=drops_key, value=1)
                        pieces_for_split = int(total_pieces * value / 100 * drops)
                        st.write(f"Split {i} pieces: {pieces_for_split}")
                        total_input += value
                        if total_input > 100:
                            st.warning("The total percentage exceeds 100%. Please adjust the values.")

                with cols[1]:
                    for j in range(st.session_state[drops_key]):
                        tn_key_j = f"tracking_number_{i}_{j+1}"
                        if tn_key_j not in st.session_state:
                            st.session_state[tn_key_j] = ''
                        _ = st.text_input(f"Tracking Number {j+1}", value=st.session_state[tn_key_j], key=tn_key_j)

                with cols[2]:
                    # CHANGED: make Letter Type options dynamic from the consolidated costs
                    letter_types = sorted(costs[st.session_state['mail_vendor']].keys())
                    for j in range(st.session_state[drops_key]):
                        lt_key_j = f"letter_type_{i}_{j+1}"
                        default_index = 0
                        if lt_key_j not in st.session_state:
                            st.session_state[lt_key_j] = letter_types[default_index]
                        _ = st.selectbox(
                            f"Letter Type {j+1}",
                            letter_types,
                            index=letter_types.index(st.session_state[lt_key_j]) if st.session_state[lt_key_j] in letter_types else default_index,
                            key=lt_key_j
                        )

                with cols[3]:
                    for j in range(st.session_state[drops_key]):
                        dd_key_j = f"{drop_date_key}_{j}"
                        if dd_key_j not in st.session_state:
                            st.session_state[dd_key_j] = pd.to_datetime('today')
                        _ = st.date_input(f"Drop Date {j+1}", key=dd_key_j)

                st.session_state['weekly_distribution'][i] = value

            if st.button('Process', key=f'process_{selected_file_name}'):
                if selected_file_name not in st.session_state:
                    st.session_state[selected_file_name] = {}

                if f'original_data_{selected_file_name}' in st.session_state:
                    st.session_state['processed_data'][selected_file_name] = st.session_state[f'original_data_{selected_file_name}'].copy()

                if input_type == "Number":
                    total_input = sum(st.session_state['weekly_distribution'].values())
                    if total_input > file_length:
                        st.warning("The total number of pieces exceeds the number of rows in the file.")
                    else:
                        combined_data = distribute_and_download(
                            df, st.session_state['weekly_distribution'], input_type,
                            file_length, selected_file_name, total_pieces, i
                        )
                        st.session_state['processed_data'][selected_file_name] = combined_data
                elif input_type == "Percentage" and abs(total_input - 100) < 1e-6:
                    combined_data = distribute_and_download(
                        df, st.session_state['weekly_distribution'], input_type,
                        file_length, selected_file_name, total_pieces, i
                    )
                    st.session_state['processed_data'][selected_file_name] = combined_data
                else:
                    st.warning("Your split percentages must total exactly 100%.")

                st.session_state[f'original_data_{selected_file_name}'] = st.session_state['processed_data'][selected_file_name].copy()

    # If all uploaded files are processed, show combined & controls
    if cleaned_files:
        file_names = list(cleaned_files.keys())
        all_files_processed = all(fn in st.session_state['processed_data'] for fn in file_names)
    else:
        all_files_processed = False

    if all_files_processed:
        combined_data_all_files = pd.concat(st.session_state['processed_data'].values())
        combined_data_all_files.columns = combined_data_all_files.columns.str.lower()
        mailer = st.session_state['mail_vendor']
        combined_data_all_files['mailer_id'] = mail_vendor_ids[mailer]

        def get_cost(row):
            letter_type = row['letter type']
            try:
                return costs[mailer][letter_type]
            except KeyError:
                st.warning(f"Letter type '{letter_type}' not found for mailer '{mailer}'. Assigning cost 0.")
                return 0.0

        combined_data_all_files['cost'] = combined_data_all_files.apply(get_cost, axis=1)
        st.dataframe(combined_data_all_files, use_container_width=True)

        breakdown = (
            combined_data_all_files
            .groupby(['file name', 'drop date', 'tracking number', 'letter type'])
            .size()
            .reset_index(name='count')
        )
        total_count = breakdown['count'].sum()
        breakdown['percentage'] = (breakdown['count'] / total_count) * 100
        breakdown['cost'] = breakdown.apply(
            lambda row: costs[st.session_state['mail_vendor']][row['letter type']] * row['count'], axis=1
        )
        st.dataframe(breakdown, use_container_width=True)

        county_breakdown = (
            combined_data_all_files
            .groupby(['county', 'letter type'])
            .size()
            .reset_index(name='count')
        )
        total_count_county = county_breakdown['count'].sum()
        county_breakdown['percentage'] = (county_breakdown['count'] / total_count_county) * 100
        county_breakdown['cost'] = county_breakdown.apply(
            lambda row: costs[st.session_state['mail_vendor']][row['letter type']] * row['count'], axis=1
        )
        st.dataframe(county_breakdown, use_container_width=True)
        st.markdown("---")

        # Per-file budgets and removal UI
        total_budget = 0
        for file_name, group in breakdown.groupby('file name'):
            file_budget = 0
            for _, row in group.iterrows():
                cost_per_piece = costs[st.session_state['mail_vendor']][row['letter type']]
                file_budget += cost_per_piece * row['count']

            st.session_state['file_budgets'][file_name] = file_budget
            st.write(f"**{file_name} Budget:** ${file_budget:.2f}")

            remove_records_number = st.number_input(
                f"Type number of rows you want to remove for {file_name}",
                min_value=0, key=f"remove_records_number_{file_name}"
            )

            # Use THIS file's dataframe, not the last selected file
            df_file = st.session_state['processed_data'][file_name]
            group_columns = st.multiselect(
                f'Select the columns that will form the groups ({file_name})',
                df_file.columns.tolist(),
                key=f"group_columns_{file_name}"
            )

            MAX_GROUPS_ALLOWED = 10_000_000
            if group_columns:
                unique_values_count = df_file[group_columns].drop_duplicates().shape[0]
                if unique_values_count > MAX_GROUPS_ALLOWED:
                    st.warning(
                        f"Selecting these columns will create {unique_values_count} unique groups "
                        f"(max {MAX_GROUPS_ALLOWED}). Choose fewer columns."
                    )
                    continue
                if 0 < remove_records_number < unique_values_count:
                    st.warning(
                        f"You must remove at least {unique_values_count} rows "
                        f"to remove one per group."
                    )
                    continue

            removed_rows_cost = 0
            remaining_rows, removed_rows = [], []
            if remove_records_number > 0 and group_columns:
                if len(df_file) >= remove_records_number:
                    try:
                        grouped = df_file.groupby(group_columns)
                        total_records = len(df_file)

                        for _, grp in grouped:
                            n_remove = int((len(grp) / total_records) * remove_records_number)
                            if len(grp) > n_remove:
                                remaining_rows.append(grp.iloc[n_remove:])
                                removed_group = grp.iloc[:n_remove]
                            else:
                                remaining_rows.append(grp.iloc[0:0])
                                removed_group = grp
                            removed_rows.append(removed_group)
                            removed_rows_cost += removed_group.apply(
                                lambda r: costs[st.session_state['mail_vendor']][r['letter type']], axis=1
                            ).sum()

                        st.write(f"Total cost of removed rows: ${removed_rows_cost:.2f}")
                        new_file_budget = file_budget - removed_rows_cost
                        st.write(f"New Cost for the file: ${new_file_budget:.2f}")
                        st.markdown("---")
                    except ValueError as e:
                        if str(e) == "No group keys passed!":
                            pass

            if st.button(f'Remove records for {file_name}', key=f'remove_records_button_{file_name}'):
                if group_columns and remove_records_number > 0 and remaining_rows:
                    updated_df = pd.concat(remaining_rows)
                    removed_df = pd.concat(removed_rows)
                    remaining_to_remove = remove_records_number - len(removed_df)
                    if remaining_to_remove > 0:
                        shuffled = updated_df.sample(frac=1).reset_index(drop=True)
                        updated_df = shuffled.iloc[remaining_to_remove:]
                    st.session_state['processed_data'][file_name] = updated_df
                    st.session_state['records_removed'] = True
                    st.rerun()
                else:
                    st.info("Nothing to remove (check your groups and remove count).")

            if st.session_state.get('records_removed'):
                st.success(f"Records successfully removed for {file_name}.")
                st.session_state['records_removed'] = False

            total_budget += file_budget

        st.markdown("---")
        total_budget = sum(st.session_state['file_budgets'].values())
        st.markdown(
            f"<h2 style='text-align: center; color: #007bff;'>Total Budget for All Files: ${max(total_budget, 0):.2f}</h2>",
            unsafe_allow_html=True
        )

        website_url = st.text_input("Enter the website URL", key="website_url")
        if st.button('Add Website Column'):
            combined_data_all_files['website'] = website_url
            st.success("Website column added successfully!")

        return_address = st.text_input("Enter the Return Address", key="return_address")
        if st.button('Add Return Address Column'):
            combined_data_all_files['return_address'] = return_address
            st.success("Return Address column added successfully!")

        client_code = st.text_input("Enter the Client Code", key="client_code")
        if st.button('Add Client Code Column'):
            combined_data_all_files['client_code'] = client_code
            st.success("Client Code column added successfully!")

        file_name_input = st.text_input("Enter the file name (without extension)", value="combined_data", key="file_name_input")
        if st.button('Download Combined Data'):
            combined_data_all_files['website'] = website_url
            combined_data_all_files['return_address'] = return_address
            combined_data_all_files['client_code'] = client_code
            download_link = create_downloadable_file(combined_data_all_files, file_name_input)
            st.session_state['download_link'] = download_link
            st.markdown(download_link, unsafe_allow_html=True)

    # Guard if user never selects a file
    if selected_file_name == "No File Selected":
        st.error("Please select a file in the sidebar to proceed.")

# ---------- WRAPPER WITH FILE UPLOADER ----------
def main():
    st.header("GGHomes Campaign Planner")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Select one or more cleaned CSVs"
    )

    if not uploaded_files:
        st.info("Upload at least one CSV to begin.")
        return

    # Prepare the dict your function expects: {filename: {"data": bytes}}
    cleaned_files = {f.name: {"data": f.read()} for f in uploaded_files}

    # Run the app
    run_gghomes_campaigns_app(cleaned_files=cleaned_files)

if __name__ == "__main__":
    main()
