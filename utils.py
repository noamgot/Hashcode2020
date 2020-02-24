import numpy as np
import os
files_names = ["a_example", "b_read_on", "c_incunabula", "d_tough_choices",
               "e_so_many_books", "f_libraries_of_the_world"]
files_names = [os.path.join("input", fn) for fn in files_names]


class HashCode2020Solver:
    def __init__(self, input_file_name):
        self.days_to_scan = 0
        self.num_books = 0
        self.num_libs = 0
        self.all_books_scores = None
        self.libs_books_matrix = None
        self.num_books_in_lib = None
        self.libs_signup_time = None
        self.libs_scan_rate = None
        self.input_file_name = input_file_name + '.txt'
        self.output_file_name = input_file_name + '_sol.txt'
        self.parse_input()

        # Solution attributes - corresponding indices!
        self.sol_libs = []
        self.sol_books_lists = []

    @staticmethod
    def split_str_to_int_list(st):
        return list(map(int, st.strip().split()))

    def parse_input(self):
        with open(self.input_file_name, 'r') as f:
            lines = f.readlines()
            num_lines = len(lines)
            if lines[-1] == '\n':
                num_lines -= 1

        self.num_books, self.num_libs, self.days_to_scan = self.split_str_to_int_list(lines[0])
        self.libs_books_matrix = np.zeros((self.num_libs, self.num_books), dtype=np.int8)
        self.libs_signup_time = np.zeros(self.num_libs, dtype=np.int32)
        self.libs_scan_rate = np.zeros(self.num_libs, dtype=np.int32)
        print(
            f"number of books: {self.num_books}, number of libraries: {self.num_libs}, scanning days: {self.days_to_scan}")
        self.all_books_scores = np.array(self.split_str_to_int_list(lines[1]), dtype=np.int32).reshape((1, -1))

        line1_idx = 2
        line2_idx = 3
        for lib_id in range(self.num_libs):
            _, t, m = self.split_str_to_int_list(lines[line1_idx])
            lib_book_ids = np.array(self.split_str_to_int_list(lines[line2_idx]), dtype=np.int32)
            # lib_book_scores = np.zeros(self.num_books, dtype=np.uint16)
            # lib_book_scores[lib_book_ids] = all_book_scores[lib_book_ids]
            self.libs_books_matrix[lib_id, lib_book_ids] = 1
            self.libs_signup_time[lib_id] = t
            self.libs_scan_rate[lib_id] = m
            line1_idx += 2
            line2_idx += 2
        self.num_books_in_lib = np.sum(self.libs_books_matrix, axis=1)

    def calculate_max_possible_books_to_scan(self):
        # number of days left to scan after signup (per library)
        scan_days_per_lib = self.days_to_scan - self.libs_signup_time
        # number of books that can be scanned in the remaining time (per library)
        books_to_scan_per_lib = scan_days_per_lib * self.libs_scan_rate
        # return the minimum between the actual number of books to the number of books that can be scanned
        result = np.minimum(books_to_scan_per_lib, self.num_books_in_lib)
        return result

    def calc_libraries_scores(self, max_books):
        scores = (np.sum(self.libs_books_matrix * self.all_books_scores, axis=1) /
                  self.num_books_in_lib) * max_books
        return scores

    def update_status_and_solution(self, all_libs_scores, max_books):
        """
        return and update the VALID_BOOKS
        """
        max_score_lib_idx = np.nanargmax(all_libs_scores)  # best library
        lib_scores = self.libs_books_matrix[max_score_lib_idx] * self.all_books_scores
        top_books_idx = np.argsort(lib_scores.flatten())
        max_books_to_scan = max_books[max_score_lib_idx]
        books_to_scan = top_books_idx[-max_books_to_scan:][::-1]  # the first best max_books of lib_id

        self.sol_libs.append(max_score_lib_idx)
        self.sol_books_lists.append(books_to_scan.tolist())

        # zero the scanned books in all the libraries
        self.libs_books_matrix[:, books_to_scan] = 0
        # zero all books in the chosen library
        self.libs_books_matrix[max_score_lib_idx, :] = 0
        # update books count
        self.num_books_in_lib = np.sum(self.libs_books_matrix, axis=1)
        # update days left
        self.days_to_scan -= self.libs_signup_time[max_score_lib_idx]

    def solve_challenge(self):
        while self.days_to_scan > 0:
            max_books = self.calculate_max_possible_books_to_scan()
            libraries_scores = self.calc_libraries_scores(max_books)
            if np.all(max_books <= 0) or np.all(np.isnan(libraries_scores)):
                break
            self.update_status_and_solution(libraries_scores, max_books)
        self.output_solutions()

    def output_solutions(self):
        num_libs = len(self.sol_libs)
        with open(self.output_file_name, 'w') as f:
            f.write(f'{num_libs}\n')
            for i, lib_idx in enumerate(self.sol_libs):
                curr_books_list = self.sol_books_lists[i]
                lib_num_books = len(curr_books_list)
                f.write(f"{lib_idx} {lib_num_books}\n{' '.join(map(str, curr_books_list))}\n")


class Hashcode2020SolverV2(HashCode2020Solver):

    def __init__(self, input_file_name):
        super().__init__(input_file_name)
        self.tmp_scan_rate = np.zeros(self.num_libs, dtype=np.int32)
        self.avg_score = np.sum(self.libs_books_matrix * self.all_books_scores, axis=1) / self.num_books_in_lib

    def find_quickest_lib(self):
        self.num_books_in_lib = np.sum(self.libs_books_matrix, axis=1)
        self.avg_score = np.sum(self.libs_books_matrix * self.all_books_scores, axis=1) / self.num_books_in_lib
        quickest_lib = np.lexsort((self.avg_score, -self.libs_scan_rate, self.libs_signup_time))[0]
        return quickest_lib

    def update_solution(self, quickest_lib):
        lib_scores = self.libs_books_matrix[quickest_lib] * self.all_books_scores
        self.days_to_scan -= self.libs_signup_time[quickest_lib]
        scan_rate = self.libs_scan_rate[quickest_lib]
        books_left_in_lib = self.num_books_in_lib[quickest_lib]
        max_books = np.minimum(books_left_in_lib, scan_rate * self.days_to_scan)
        books_to_ship = np.argsort(lib_scores)[-max_books:][::-1]

        self.sol_libs.append(quickest_lib)
        self.sol_books_lists.append(list(books_to_ship.flatten()))
        self.libs_books_matrix[:, books_to_ship] = 0
        self.libs_books_matrix[quickest_lib, :] = 0
        self.num_books_in_lib = np.sum(self.libs_books_matrix, axis=1)

        self.libs_scan_rate[quickest_lib] = np.min(self.libs_scan_rate)
        self.libs_signup_time[quickest_lib] = np.max(self.libs_signup_time)

    def solve_challenge(self):
        while self.days_to_scan > 0 and np.any(self.num_books_in_lib != 0):
            quickest_lib = self.find_quickest_lib()
            self.update_solution(quickest_lib)
            if np.all(self.libs_signup_time == self.libs_signup_time[0]) and np.all(
                    self.libs_scan_rate == self.libs_scan_rate[0]):
                break
        self.output_solutions()
