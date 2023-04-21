# explainability-challenges

## Testing

The test suite is executable via the `pytest` package.
First, install `pytest` with 
```bash
pip install pytest
```
Then, install the package locally for development purposes by running
```bash
pip install -e .
```
in the root directory of the project.
Finally, run the test suite with
```bash
pytest
```
To run only fast tests, run
```bash
pytest -m "not slow"
```


For Windows devices, if the script hangs due to `kaleido`, run
```bash
pip uninstall kaleido
pip install kaleido==0.1.0.post1
```