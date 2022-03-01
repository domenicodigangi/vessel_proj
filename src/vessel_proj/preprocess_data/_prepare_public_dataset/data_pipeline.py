from prefect import flow

from ... import create_raw_data_local_artifacts, clean_visits, create_and_log_edge_list_from_visits, aggregated_graph_from_edge_list

# THIS PIPELINE HAS NEVER BEEN TESTED AND IT ASSUMES THAT THE RAW VISITS' LIST IS ALREADY IN DATA/RAW

@flow
def main():
    create_raw_data_local_artifacts.main()
    clean_visits.main()
    create_and_log_edge_list_from_visits.main()
    aggregated_graph_from_edge_list.main()
    
