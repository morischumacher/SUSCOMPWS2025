

### Python

Install virtualenv

```console
$ python3 -m venv venv
```
For VSCode select kernel from the venv folder

Setup virtualenv and activate it

```console
$ source venv.sh

(To deactivate the virtualenv execute the following command):
$ deactivate
```

Install strip package to not run into many merge conflict
```console
$ pip install nbstripout
nbstripout --install
```

For outputs we want to commit we can add this metadata to a cell:
```
Set the "keep_output": true metadata on the cell. To do this, select the "Edit Metadata" Cell Toolbar, and then use the "Edit Metadata" button on the desired cell to enter something like:

{
  "keep_output": true,
}
```