import contextlib
import io
import sys




class OutputCapture:
    def __init__(self, verbose):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO())
            self.out = self.capture.__enter__()

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            if exc_type is not None:
                print(f'Failed with output:\n {self.out.getvalue()}', )
        sys.stdout.flush()
