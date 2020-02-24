from utils import HashCode2020Solver
from utils import files_names

if __name__ == '__main__':
    print("Solving Hashcode2020 challenge...")
    for fn in files_names:
        print("************************************************************************************")
        print("Solving input file:", fn)
        try:
            solver = HashCode2020Solver(fn)
            solver.solve_challenge()
            print(f"Done - solution can be found in {solver.output_file_name}")
        except Exception as e:
            print("Caught an exception for file:", fn)
            print(e)