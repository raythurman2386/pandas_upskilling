import pandas as pd


def read_csv_example():
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")


def group_by_example():
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
    grouped = data.groupby('sex')
    print(grouped.size())
    print(grouped['total_bill'].mean())


def describe_example():
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
    print(data.describe())


def correlation_example():
    data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
    print(data.corr())


if __name__ == '__main__':
    functions = {
        '1': read_csv_example,
        '2': group_by_example,
        '3': describe_example,
        '4': correlation_example
    }

    print("Choose a function to run:")
    print("1. Read CSV example")
    print("2. Group by example")
    print("3. Describe example")
    print("4. Correlation example")

    choice = input("Enter your choice (1-4): ")

    if choice in functions:
        functions[choice]()
    else:
        print("Invalid choice. Please run the script again and select a number between 1 and 4.")