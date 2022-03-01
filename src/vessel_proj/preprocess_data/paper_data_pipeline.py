from prefect import flow

import create_and_log_centralities
import create_and_log_wpi_ports_feat
import download_and_log_graph_data

# THIS PIPELINE HAS NEVER BEEN TESTED AND IT ASSUMES THAT THE RAW VISITS' LIST IS ALREADY IN DATA/RAW

@flow
def main():
    
    download_and_log_graph_data.main()
    create_and_log_centralities.main()
    create_and_log_wpi_ports_feat.main()
    
