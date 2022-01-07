from vessel_proj.data import get_data_path, shaper_slow

input_file = get_data_path() / "raw" /'visits-augmented.csv'
output_file = get_data_path() / "raw" /'voyage_links.csv'

# df_vis = pd.read_csv(input_file)
# df_ports = pd.read_csv(get_data_path() / "raw" / "ports-1000.csv")
# shaper_slow(input_file, output_file, nrows = 100000)

def main():
    shaper_slow(input_file, output_file)

if __name__ == "__main__":
    main()
 