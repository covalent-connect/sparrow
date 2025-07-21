from settings import aws_entrypoint

if __name__ == "__main__":
    print("Running functions")
    aws_entrypoint.run_functions()

# python -m sparrow_parse.testing.run_functions