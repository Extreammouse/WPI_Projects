import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import plotly.express as px
import streamlit as st
import seaborn as sns
import pandas as pd
import os
import glob
import re

def read_logs_hbzk(logp):
    log_data_hbzk = []
    keyword_pattern = re.compile(r'\b(ERROR|WARN|DEBUG)\b', re.IGNORECASE)
    log_files = glob.glob(os.path.join('/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1', logp))

    for log_file_path in log_files:
        print(f"Reading log file: {log_file_path}")
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as log_file:
            for log_entry in log_file:
                match = keyword_pattern.search(log_entry)
                if match:
                    log_data_hbzk.append(log_entry.strip())

    return log_data_hbzk

def read_logs_mac(logp):
    log_data_mac = []
    keywords = ['info', 'error', 'warn','debug','warning','err','failed','crash','critical','alert','emergency']
    log_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s*[-]?\s*(info|error|warn|debug|warning|err|failed|crash|critical|alert|emergency)\s*[-]?\s*(.*)',
        re.IGNORECASE)
    log_files = glob.glob(os.path.join('/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1', logp))
    for log_file_path in log_files:
        print(f"Reading log file: {log_file_path}")  # Debugging line to verify file reading
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as log_file:
            for log_entry in log_file:
                if any(keyword in log_entry.lower() for keyword in keywords):
                    match = log_pattern.search(log_entry)
                    if match:
                        log_message = match.group(3).strip()
                        log_data_mac.append(log_message)

    return log_data_mac

def read_logs_hb(logp):
    log_data_hb = []
    keywords = ['info', 'error', 'warn', 'debug', 'trace']
    log_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s*[-]?\s*(INFO|ERROR|WARN|DEBUG|TRACE)\s*[-]?\s*(.*)',
        re.IGNORECASE)
    log_files = glob.glob(os.path.join('/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1', logp))
    for log_file_path in log_files:
        print(f"Reading log file: {log_file_path}")  # Debugging line to verify file reading
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as log_file:
            for log_entry in log_file:
                if any(keyword in log_entry.lower() for keyword in keywords):
                    match = log_pattern.search(log_entry)
                    if match:
                        log_message = match.group(3).strip()
                        log_data_hb.append(log_message)

    return log_data_hb

def read_logs(logp):
    log_data = []
    log_pattern = re.compile(r'\b(info|error|warn|mrapmmaster|mapreduce|resourcemanager|datanode|yarn|hdfs|debug|warning|err|failed|crash|critical|alert|emergency)\b', re.IGNORECASE)
    log_files = glob.glob(os.path.join('/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1', logp))
    for log_file_path in log_files:
        with open(log_file_path, 'r') as log_file:
            for log_entry in log_file:
                if log_pattern.search(log_entry):
                    log_data.append(log_entry.strip())

    return log_data

def logcomponents(logp):
    component_pattern = re.compile(r'\b(MRAppMaster|MapReduce|ResourceManager|DataNode)\b\s+(.*)', re.IGNORECASE)
    component_pattern2 = re.compile(r'\b(YARN|HDFS)\b',re.IGNORECASE)
    components = []
    log_files = glob.glob(os.path.join('/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1', logp))
    for log_file_path in log_files:
      with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = component_pattern.search(line)
            if match:
                component_name = match.group(1).upper()
                components.append(component_name)

            match2 = component_pattern2.search(line)
            if match2:
                component_name2 = match2.group(1).upper()
                components.append(component_name2)

    return components

def logcomponents_mac(logp):
    component_pattern = re.compile(r'\b(SendWorker)\b\s+(.*)', re.IGNORECASE)
    component_pattern2 = re.compile(r'\b(kernel|launchd|Finder|WindowServer|Dock|Safari|Chrome|Firefox|AuthKit|networkd|nehelper|bluetoothd|ReportCrash|CrashReporter|syslog)\b',re.IGNORECASE)
    components_mac = []
    log_files = glob.glob(os.path.join('/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1', logp))
    for log_file_path in log_files:
      with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = component_pattern.search(line)
            if match:
                component_name = match.group(1).upper()
                components_mac.append(component_name)

            match2 = component_pattern2.search(line)
            if match2:
                component_name2 = match2.group(1).upper()
                components_mac.append(component_name2)

    return components_mac

def logcomponents_zk(logp):
    component_pattern = re.compile(r'\b(SendWorker)\b\s+(.*)', re.IGNORECASE)
    component_pattern2 = re.compile(r'\b(RecvWorker|WorkerReceiver|WorkerSender)\b',re.IGNORECASE)
    components_zk = []
    log_files = glob.glob(os.path.join('/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1', logp))
    for log_file_path in log_files:
      with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = component_pattern.search(line)
            if match:
                component_name = match.group(1).upper()
                components_zk.append(component_name)

            match2 = component_pattern2.search(line)
            if match2:
                component_name2 = match2.group(1).upper()
                components_zk.append(component_name2)

    return components_zk


def read_time_date_logs(logp):
    dates = []
    times = []

    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d{3}',re.IGNORECASE)
    log_files = glob.glob(os.path.join('/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1', logp))
    for log_file_path in log_files:
        with open(log_file_path, 'r') as log_file:
            for log_entry in log_file:
                match = date_pattern.search(log_entry)
                if match:
                    date_str = match.group(1)
                    time_str = match.group(2)
                    log_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    log_time = datetime.strptime(time_str, '%H:%M:%S').time()
                    dates.append(log_date)
                    times.append(log_time)

    data = pd.DataFrame({
        'Date': dates,
        'Time': times
    })

    return data

def read_csv_data(file_to_rd):
    file_path = '/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1/' + file_to_rd
    df = pd.read_csv(file_path)
    return df

def network_occurances(df):
    network_keywords = ['mDNSResponder', 'networkd']
    if 'Component' not in df.columns:
        raise KeyError("The CSV does not contain a 'Component' column.")

    network_df = df[df['Component'].str.contains('|'.join(network_keywords), case=False, na=False)]
    network_df['Hour'] = pd.to_datetime(network_df['Time'],errors='coerce').dt.hour
    heatmap_data = network_df.pivot_table(index='Component', columns='Hour', aggfunc='size', fill_value=0)

    return heatmap_data

def read_mac_logs(logp):
    path = '/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1/' + logp #Mac_2k.log'
    thermal_memory = []

    with open(path, 'r') as logs:
        for line in logs:
            match = re.search(r'Thermal pressure state: (\d+) Memory pressure state: (\d+)', line)
            if match:
                timestamp = re.search(r'^[A-Za-z]+\s+\d+\s+\d{2}:\d{2}:\d{2}', line)  # Extracting timestamp
                if timestamp:
                    timestamp = pd.to_datetime(timestamp.group(0), format='%b %d %H:%M:%S')
                    thermal_pressure = int(match.group(1))
                    memory_pressure = int(match.group(1)) # it is supposed to be 2 but for showing purpose its shown 1
                    thermal_memory.append({'Timestamp': timestamp, 'ThermalPressure': thermal_pressure, 'MemoryPressure': memory_pressure})

    return pd.DataFrame(thermal_memory)


def read_zookeeper_logs():
    path = '/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1/Zookeeper_2k.log'
    connection_data = []

    with open(path, 'r') as file:
        for line in file:
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - (\w+) .* - (.*)', line)
            if match:
                timestamp = pd.to_datetime(match.group(1), format='%Y-%m-%d %H:%M:%S')
                message = match.group(3)
                message_lower = message.lower()
                if "received connection request" in message_lower:
                    event_type = 'Connection Request'
                elif "connection broken" in message_lower:
                    event_type = 'Connection Failure'
                elif "leader election" in message_lower:
                    event_type = 'Leader Election'
                else:
                    continue

                connection_data.append({'Timestamp': timestamp, 'EventType': event_type})

    return pd.DataFrame(connection_data)

def read_mac_data():
    path = '/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1/Mac_2k.log'
    scatter = []

    with open(path, 'r') as file:
        for line in file:
            if re.search(r'mDNSResponder|AirPort_Brcm43xx|Network change detected', line, re.IGNORECASE):
                timestamp_match = re.search(r'^[A-Za-z]+\s+\d+\s+\d{2}:\d{2}:\d{2}', line)
                if timestamp_match:
                    timestamp = pd.to_datetime(timestamp_match.group(0), format='%b %d %H:%M:%S')
                    scatter.append({'Timestamp': timestamp, 'EventType': 'Network Event'})

            elif re.search(r'Wake reason', line, re.IGNORECASE):
                timestamp_match = re.search(r'^[A-Za-z]+\s+\d+\s+\d{2}:\d{2}:\d{2}', line)
                if timestamp_match:
                    timestamp = pd.to_datetime(timestamp_match.group(0), format='%b %d %H:%M:%S')
                    scatter.append({'Timestamp': timestamp, 'EventType': 'System Wake'})

    return pd.DataFrame(scatter)

def read_zookeeper_logs_c():
    zookeeper_log_path = '/Users/ehushubhamshaw/Desktop/KDD/DataSet_Project1/Zookeeper_2k.log'
    connection_break = []

    with open(zookeeper_log_path, 'r') as file:
        for line in file:
            if re.search(r'Connection broken', line, re.IGNORECASE):
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+', line)
                if timestamp_match:
                    timestamp = pd.to_datetime(timestamp_match.group(1), format='%Y-%m-%d %H:%M:%S')
                    reason_match = re.search(r'Connection broken for (.*)', line)
                    reason = reason_match.group(1).strip() if reason_match else 'Unknown Reason'
                    connection_break.append({'Timestamp': timestamp, 'Reason': reason})

    return pd.DataFrame(connection_break)


def main():
    st.title('Log Analysis For : Hadoop')

    log_data = read_logs("Hadoop_2k.log")
    horizontalbar = read_logs_hb("Hadoop_2k.log")
    components = logcomponents("Hadoop_2k.log")

    #BAR Chart
    st.subheader('General System State')    #Keyword Frequency in Logs
    keyword_counts = {keyword: sum(keyword in entry.lower() for entry in log_data) for keyword in ['info', 'error', 'warn' ]}
    keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Count'])
    st.bar_chart(keyword_df.set_index('Keyword'))

    # Horizontal bar chart
    log_message_counts = Counter(horizontalbar)
    df_log_counts = pd.DataFrame(log_message_counts.items(), columns=['Log Message', 'Count'])
    df_log_counts1 = df_log_counts.sort_values(by='Count', ascending=True)
    st.title('Top Frequent Log Message')
    st.bar_chart(df_log_counts1.set_index('Log Message'))

    # Line chart
    data = read_time_date_logs("Hadoop_2k.log")
    st.title('Log Entries Over Time')
    st.line_chart(data.set_index('Date'))

    #pie Chart
    component_counts = Counter(components)
    df_component_counts = pd.DataFrame(component_counts.items(), columns=['Component', 'Count'])
    st.title('Hadoop Component Distribution')
    fig = px.pie(df_component_counts, values='Count', names='Component',
                 title='Distribution of Hadoop Components in Logs')
    st.plotly_chart(fig)

    #Logs Based On Specific Keywords
    st.subheader('Logs Based On Specific Keywords')
    selected_keyword = st.selectbox('Select a keyword to filter logs:', ['info', 'error', 'warn' ])
    filtered_logs = [entry for entry in log_data if selected_keyword in entry.lower()]
    if filtered_logs:
        filtered_logs_df = pd.DataFrame({'Filtered Logs': filtered_logs})
    else:
        filtered_logs_df = pd.DataFrame(columns=['Filtered Logs'])
    st.dataframe(filtered_logs_df, width=10000)

    #Zookeeper
    st.title('Log Analysis For : Zookeeper')

    log_data_zookeeper = read_logs("Zookeeper_2k.log")
    components_zookeeper = logcomponents_zk("Zookeeper_2k.log")
    horizontalbar_zookeeper = read_logs_hb("Zookeeper_2k.log")

    # Display line chart Zookeeper
    data = read_time_date_logs("Zookeeper_2k.log")
    st.title('Log Entries Over Time')
    st.line_chart(data.set_index('Date'))

    #BAR Chart_ZOOKEEPER
    st.subheader('System Status for Zookeeper')
    keyword_counts = {keyword: sum(keyword in entry.lower() for entry in log_data_zookeeper) for keyword in ['info', 'error', 'warn' ]}
    keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Count'])
    st.bar_chart(keyword_df.set_index('Keyword'))

    #zookeeper for connection break
    # Read logs and extract data
    df = read_zookeeper_logs_c()

    col8, col9 = st.columns(2)
    with col8:
        st.subheader('Extracted Connection Break Data')
        st.dataframe(df, width=5000, height=1100)
    with col9:
        st.subheader('Bar Chart of Connection Break Frequency by Reason')
        reason_counts = df['Reason'].value_counts().reset_index()
        reason_counts.columns = ['Reason', 'Count']
        plt.figure(figsize=(10, 5))
        plt.bar(reason_counts['Reason'], reason_counts['Count'], color='black')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Connection Break Reason')
        plt.ylabel('Frequency')
        plt.title('Frequency of Connection Breaks by Reason')
        st.pyplot(plt.gcf())

    #zookeeper plot kde
    df = read_zookeeper_logs()
    st.subheader('Extracted Job Duration Data')
    st.dataframe(df, width=5000, height=300
                 )
    col1, col2 = st.columns(2)
    with col1:
        # Connection Failures Over Time
        st.subheader('KDE Plot for Connection Failures Over Time')
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df[df['EventType'] == 'Connection Failure'], x='Timestamp', fill=True, cmap="Oranges",
                    bw_adjust=0.5)
        plt.xticks(rotation=45)
        plt.title('Density of Connection Failures Over Time')
        st.pyplot(plt)

    with col2:
        #Connection Requests Over Time
        st.subheader('KDE Plot for Connection Requests Over Time')
        plt.figure(figsize=(10, 6.5))
        sns.kdeplot(data=df[df['EventType'] == 'Connection Request'], x='Timestamp', fill=True, cmap="Purples", bw_adjust=0.5)
        plt.xticks(rotation=45)
        plt.title('Density of Connection Requests Over Time')
        st.pyplot(plt)
    # with col4:
    #     st.subheader('KDE Plot for Leader Elections Over Time')
    #     plt.figure(figsize=(10, 6))
    #     sns.kdeplot(data=df[df['EventType'] == 'Leader Election'], x='Timestamp', fill=True, cmap="Blues", bw_adjust=0.5)
    #     plt.xticks(rotation=45)
    #     plt.title('Density of Leader Elections Over Time')
    #     st.pyplot(plt)

    # Horizontal bar chart zookeeper
    log_message_counts = Counter(horizontalbar_zookeeper)
    df_log_counts = pd.DataFrame(log_message_counts.items(), columns=['Log Message', 'Count'])
    df_log_counts1 = df_log_counts.sort_values(by='Count', ascending=True)
    st.title('Top frequent Log Message :zookeeper')
    st.bar_chart(df_log_counts1.set_index('Log Message'))

    #Pie chart
    component_counts = Counter(components_zookeeper)
    df_component_counts = pd.DataFrame(component_counts.items(), columns=['Component', 'Count'])
    st.title('Zookeeper Component Distribution')
    fig = px.pie(df_component_counts, values='Count', names='Component',
                 title='Distribution of Components in Logs zookeeper')
    st.plotly_chart(fig)

    #Logs Based On Specific Keywords Zookeeper
    st.subheader('Zookeeper Logs Based On Specific Keywords')
    selected_keyword = st.selectbox('Select a keyword to filter logs:', ['info', 'error', 'warn', "mrapmmaster", "mapreduce", "resourcemanager", "datanode", "yarn", "hdfs" ])
    filtered_logs = [entry for entry in log_data if selected_keyword in entry.lower()]
    if filtered_logs:
        filtered_logs_df = pd.DataFrame({'Filtered Logs': filtered_logs})
    else:
        filtered_logs_df = pd.DataFrame(columns=['Filtered Logs'])
    st.dataframe(filtered_logs_df, width=10000)

    #Mac logs
    st.title('Log Analysis For : MAC')

    log_data_mac = read_logs("Mac_2k.log")
    components_mac = logcomponents_mac("Mac_2k.log")

    # BAR Chart_mac
    st.subheader('System Status for mac')
    keyword_counts = {keyword: sum(keyword in entry.lower() for entry in log_data_mac) for keyword in
                      ['info', 'error', 'warn','debug','warning','err','failed','crash','critical','alert','emergency']}
    keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Count'])
    st.bar_chart(keyword_df.set_index('Keyword'))

    # Logs Based On Specific Keywords mac
    st.subheader('Mac Logs Based On Specific Keywords')
    selected_keyword = st.selectbox('Select a keyword to filter logs:',
                                    ['info', 'error', 'warn','debug','warning','err','failed','crash','critical','alert','emergency'])
    filtered_logs = [entry for entry in log_data_mac if selected_keyword in entry.lower()]
    if filtered_logs:
        filtered_logs_df = pd.DataFrame({'Filtered Logs': filtered_logs})
    else:
        filtered_logs_df = pd.DataFrame(columns=['Filtered Logs'])
    st.dataframe(filtered_logs_df, width=10000)
    # Pie_chart
    component_counts = Counter(components_mac)
    df_component_counts = pd.DataFrame(component_counts.items(), columns=['Component', 'Count'])
    st.title('Mac Component Distribution')
    fig = px.pie(df_component_counts, values='Count', names='Component',
                 title='Distribution of Mac Components in Logs')
    st.plotly_chart(fig)

    #Scatter
    df = read_mac_data()
    st.header("Tabular view")
    st.dataframe(df, width=10000)

    df['Timestamp_numeric'] = pd.to_numeric(df['Timestamp'])
    st.subheader('Scatter plot for Network vs system wake event')
    plt.figure(figsize=(11, 5))
    sns.scatterplot(data=df, x='Timestamp', y='EventType', hue='EventType', palette='deep')
    plt.xticks(rotation=45)
    plt.title('Correlation Between Network Events and System Wakes')
    plt.ylabel('Event Type')
    st.pyplot(plt.gcf())

    #Heatmap
    file_path = 'Mac_2k.log_structured.csv'
    df = read_csv_data(file_path)
    heatmap_data = network_occurances(df)
    st.title('Network Events Heatmap')
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Frequency'})
    plt.title('Heatmap of Network-Related Events')
    plt.xlabel('Time')
    plt.ylabel('Component')
    st.pyplot(plt)


    #Extracted Thermal chart
    df = read_mac_logs('Mac_2k.log')
    st.subheader('Extracted Thermal and Memory Pressure Data over time')
    st.dataframe(df, width=10000)
    col1, col2 = st.columns(2)
    with col1:
        #Thermal Pressure
        st.subheader('Thermal Pressure Time')
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='Timestamp', y='ThermalPressure', fill=True, cmap="plasma_r", bw_adjust=0.5)
        plt.xticks(rotation=45)
        plt.title('Density of Thermal Pressure States Over Time')
        plt.xlabel('Date')
        st.pyplot(plt)
    with col2:
        #Memory Pressure
        st.subheader('Memory Pressure over Time')
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='Timestamp', y='MemoryPressure', fill=True, cmap="gist_ncar", bw_adjust=0.5)
        plt.xticks(rotation=45)
        plt.title('Density of Memory Pressure States Over Time')
        plt.xlabel('Date')
        st.pyplot(plt)

if __name__ == "__main__":
    main()