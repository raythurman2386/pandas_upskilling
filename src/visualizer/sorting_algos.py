import matplotlib.pyplot as plt
import numpy as np
import time


class SortingAlgorithm:
    def get_generator(self, arr):
        raise NotImplementedError("Subclasses must implement this method")


# Bubble Sort implementation
class BubbleSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.bubble_sort(arr)

    def bubble_sort(self, arr):
        n = len(arr)
        for i in range(n - 1):
            for j in range(n - i - 1):
                yield (arr.copy(), [j, j + 1])
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    yield (arr.copy(), [j, j + 1])


# Selection Sort implementation
class SelectionSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.selection_sort(arr)

    def selection_sort(self, arr):
        n = len(arr)
        for i in range(n - 1):
            min_idx = i
            for j in range(i + 1, n):
                yield (arr.copy(), [i, j])
                if arr[j] < arr[min_idx]:
                    min_idx = j
            yield (arr.copy(), [i, min_idx])
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            yield (arr.copy(), [i, min_idx])


# Insertion Sort implementation
class InsertionSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.insertion_sort(arr)

    def insertion_sort(self, arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                yield (arr.copy(), [j, j + 1])
                j -= 1
            arr[j + 1] = key
            yield (arr.copy(), [j + 1])


# Merge Sort implementation
class MergeSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.merge_sort(arr, 0, len(arr) - 1)

    def merge_sort(self, arr, start, end):
        if start < end:
            mid = (start + end) // 2
            yield from self.merge_sort(arr, start, mid)
            yield from self.merge_sort(arr, mid + 1, end)
            left = arr[start : mid + 1].copy()
            right = arr[mid + 1 : end + 1].copy()
            i = j = 0
            k = start
            while i < len(left) and j < len(right):
                yield (arr.copy(), [k, start + i, mid + 1 + j])
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1
                yield (arr.copy(), [k])
            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1
                yield (arr.copy(), [k])
            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1
                yield (arr.copy(), [k])


# Quick Sort implementation
class QuickSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.quick_sort(arr, 0, len(arr) - 1)

    def quick_sort(self, arr, low, high):
        if low < high:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                yield (arr.copy(), [j, high])
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    yield (arr.copy(), [i, j])
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            yield (arr.copy(), [i + 1, high])
            yield from self.quick_sort(arr, low, i)
            yield from self.quick_sort(arr, i + 2, high)


# Heap Sort implementation
class HeapSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.heap_sort(arr)

    def heap_sort(self, arr):
        n = len(arr)
        # Build heap
        for i in range(n // 2 - 1, -1, -1):
            yield from self.heapify(arr, n, i)
        # Extract elements
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            yield (arr.copy(), [0, i])  # Highlight swap
            yield from self.heapify(arr, i, 0)

    def heapify(self, arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            yield (arr.copy(), [i, largest])  # Highlight swap
            yield from self.heapify(arr, n, largest)


# Shell Sort implementation
class ShellSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.shell_sort(arr)

    def shell_sort(self, arr):
        n = len(arr)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    yield (arr.copy(), [j, j - gap])  # Highlight comparison
                    j -= gap
                arr[j] = temp
                yield (arr.copy(), [j])  # Highlight insertion
            gap //= 2


# Radix Sort implementation (for non-negative integers)
class RadixSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.radix_sort(arr)

    def radix_sort(self, arr):
        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            yield from self.counting_sort(arr, exp)
            exp *= 10

    def counting_sort(self, arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
        for i in range(n):
            arr[i] = output[i]
        yield (arr.copy(), [])  # Yield after each digit's processing


# Bucket Sort implementation (for integers)
class BucketSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.bucket_sort(arr)

    def bucket_sort(self, arr):
        n = len(arr)
        buckets = [[] for _ in range(n)]
        # Distribute elements into buckets
        for num in arr:
            index = num // 5  # Assuming multiples of 5
            buckets[min(index, n - 1)].append(num)
        # Sort each bucket and yield
        for i, bucket in enumerate(buckets):
            bucket.sort()
            yield (arr.copy(), [])  # Yield after sorting each bucket
        # Concatenate buckets
        index = 0
        for bucket in buckets:
            for num in bucket:
                arr[index] = num
                index += 1
        yield (arr.copy(), [])  # Final sorted array


# Tim Sort implementation
class TimSort(SortingAlgorithm):
    def get_generator(self, arr):
        """
        Generates steps for TimSort, yielding (arr.copy(), highlight_indices) at each step.
        """
        n = len(arr)
        min_run = self.calc_min_run(n)
        run_ends = []
        i = 0
        while i < n:
            run_start = i
            if i < n - 1:
                if arr[i] <= arr[i + 1]:
                    # Ascending run
                    while i < n - 1 and arr[i] <= arr[i + 1]:
                        i += 1
                else:
                    # Descending run
                    while i < n - 1 and arr[i] > arr[i + 1]:
                        i += 1
                    # Reverse the descending run to make it ascending
                    arr[run_start:i + 1] = arr[run_start:i + 1][::-1]
                    yield arr.copy(), list(range(run_start, i + 1))
            run_end = i
            # If the run is shorter than min_run, extend it
            if run_end - run_start + 1 < min_run and i < n - 1:
                extension_end = min(n - 1, run_start + min_run - 1)
                # Sort the extended run with insertion sort
                for step in self.insertion_sort_gen(arr, run_start, extension_end):
                    yield step
                run_end = extension_end
            run_ends.append(run_end + 1)
            i = run_end + 1
        # Merge runs iteratively
        while len(run_ends) > 1:
            left = 0 if len(run_ends) == 1 else run_ends[0]
            mid = run_ends[0] - 1
            right = run_ends[1] - 1
            for step in self.merge_gen(arr, left, mid, right):
                yield step
            run_ends = [right + 1] + run_ends[2:]
        return arr

    def calc_min_run(self, n):
        """
        Calculates the minimum run size for TimSort based on the array length.

        Parameters:
            n (int): Length of the array.

        Returns:
            int: Minimum run size.
        """
        MIN_MERGE = 32
        r = 0
        while n >= MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def insertion_sort_gen(self, arr, left, right):
        """
        Generates steps for insertion sort on arr[left:right+1].

        Parameters:
            arr (list): The array to sort.
            left (int): Starting index of the subarray.
            right (int): Ending index of the subarray.

        Yields:
            tuple: (array_copy, highlight_indices).
        """
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                yield arr.copy(), [j, j + 1]
                j -= 1
            arr[j + 1] = key
            yield arr.copy(), [left, i]

    def merge_gen(self, arr, left, mid, right):
        """
        Generates steps for merging arr[left:mid+1] and arr[mid+1:right+1].

        Parameters:
            arr (list): The array to merge.
            left (int): Start index of the first run.
            mid (int): End index of the first run.
            right (int): End index of the second run.

        Yields:
            tuple: (array_copy, highlight_indices).
        """
        left_arr = arr[left:mid + 1].copy()
        right_arr = arr[mid + 1:right + 1].copy()
        i = j = 0
        k = left
        while i < len(left_arr) and j < len(right_arr):
            # Highlight the elements being compared and the write position
            yield arr.copy(), [k, left + i, mid + 1 + j]
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            k += 1
            # Highlight the write position after assignment
            yield arr.copy(), [k - 1]
        while i < len(left_arr):
            arr[k] = left_arr[i]
            i += 1
            k += 1
            yield arr.copy(), [k - 1]
        while j < len(right_arr):
            arr[k] = right_arr[j]
            j += 1
            k += 1
            yield arr.copy(), [k - 1]


# Cocktail Sort implementation
class CocktailSort(SortingAlgorithm):
    def get_generator(self, arr):
        return self.cocktail_sort(arr)

    def cocktail_sort(self, arr):
        n = len(arr)
        swapped = True
        start = 0
        end = n - 1
        while swapped:
            swapped = False
            # Forward pass
            for i in range(start, end):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
                    yield (arr.copy(), [i, i + 1])
            if not swapped:
                break
            swapped = False
            end -= 1
            # Backward pass
            for i in range(end - 1, start - 1, -1):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
                    yield (arr.copy(), [i, i + 1])
            start += 1


# Visualizer class (with update to handle no highlights)
class Visualizer:
    def __init__(self, arr):
        self.arr = arr
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.bars = None
        self.results = []

    def update(self, step):
        arr_state, highlight_indices = step
        for bar, height in zip(self.bars.patches, arr_state):
            bar.set_height(height)
        if highlight_indices:  # Only highlight if indices are provided
            for idx in highlight_indices:
                if idx < len(self.bars.patches):
                    self.bars.patches[idx].set_color("red")
        self.ax.set_title(f"{self.algorithm_name} - Step {self.step_count}")
        plt.draw()

    def run(self, generator, initial_arr, algorithm_name):
        self.algorithm_name = algorithm_name
        self.ax.clear()
        self.bars = self.ax.bar(
            range(len(initial_arr)), initial_arr, color="blue", width=0.8
        )
        self.ax.set_xticks(range(len(initial_arr)))
        self.ax.set_xticklabels([])
        self.ax.set_yticks([])
        self.step_count = 0
        start_time = time.perf_counter()
        for step in generator:
            self.step_count += 1
            self.update(step)
            plt.pause(0.01)
            for bar in self.bars.patches:
                bar.set_color("blue")
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        print(f"{algorithm_name} took {time_taken:.4f} seconds")
        self.results.append(
            {"algorithm": algorithm_name, "time": time_taken, "steps": self.step_count}
        )

    def show_summary(self):
        if not self.results:
            return
        algorithms = [res["algorithm"] for res in self.results]
        times = [res["time"] for res in self.results]
        steps = [res["steps"] for res in self.results]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.bar(algorithms, times, color="skyblue")
        ax1.set_title("Time Taken by Each Algorithm")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels(algorithms, rotation=45, ha="right")
        ax2.bar(algorithms, steps, color="lightgreen")
        ax2.set_title("Step Count by Each Algorithm")
        ax2.set_ylabel("Number of Steps")
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels(algorithms, rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


# Main function
def main():
    algorithms = {
        "Bubble Sort": BubbleSort(),
        "Selection Sort": SelectionSort(),
        "Insertion Sort": InsertionSort(),
        "Merge Sort": MergeSort(),
        "Quick Sort": QuickSort(),
        "Heap Sort": HeapSort(),
        "Shell Sort": ShellSort(),
        "Radix Sort": RadixSort(),
        "Bucket Sort": BucketSort(),
        "Tim Sort": TimSort(),
        "Cocktail Sort": CocktailSort(),
    }

    while True:
        try:
            size = int(input("Enter the array size (e.g., 10-50): "))
            if 1 <= size <= 100:
                break
            print("Please enter a size between 1 and 100.")
        except ValueError:
            print("Please enter a valid integer.")

    multiples = np.arange(1, size + 1) * 5
    np.random.shuffle(multiples)
    arr = multiples
    print(f"Initial array: {arr}")

    # Initialize visualizer
    visualizer = Visualizer(arr)
    for name, algorithm in algorithms.items():
        print(f"Starting {name}...")
        arr_copy = arr.copy()  # Reset array for each algorithm
        generator = algorithm.get_generator(arr_copy)
        visualizer.run(generator, arr_copy, name)
        print(f"{name} completed.\n")

    # Close the visualization window and show summary
    plt.close(visualizer.fig)
    visualizer.show_summary()


if __name__ == "__main__":
    main()
