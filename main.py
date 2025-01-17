import pandas as pd
import geopandas as gpd


def read_csv_example():
    """Basic read CSV example"""
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")


def group_by_example():
    """Group by example"""
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
    grouped = data.groupby('sex')
    print(grouped.size())
    print(grouped['total_bill'].mean())


def describe_example():
    """Describe example"""
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
    print(data.describe())


def correlation_example():
    """Correlation example"""
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
    print(data.corr())


def states_example():
    """Geopandas example"""
    data = gpd.read_file('states_provinces.shp')
    data["area"] = data["geometry"].area
    print(data.head(5))
    fig, ax = plt.subplots(figsize=(15, 10))
    data.plot(ax=ax, color='none', edgecolor='blue')
    ax.set_title('Antarctic ice shelves')
    plt.show()


if __name__ == '__main__':
    functions = {
        1: read_csv_example,
        2: group_by_example,
        3: describe_example,
        4: correlation_example,
        5: states_example
    }

    while True:
        print("\nAvailable examples:")
        for num, func in functions.items():
            print(f"{num}. {func.__doc__}")
        print("0. Exit")

        choice = input("Enter the number of the example you want to run (0 to exit): ")
        if choice == '0':
            break
        elif choice.isdigit() and int(choice) in functions:
            functions[int(choice)]()
        else:
            print("Invalid choice. Please try again.")
