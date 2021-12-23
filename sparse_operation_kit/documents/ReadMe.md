# Documents #
Want to find more about SparseOperationKit, see our [SparseOperationKit documents](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/v1.0.1/index.html).

## Add documents for SparseOperationKit ##
+ Install the required tools and extensions
```shell
$ pip install sphinx recommonmark nbsphinx sphinx_rtd_theme sphinx_multiversion sphinx_markdown_tables myst-parser linkify-it-py
```

+ Configure sphinx
```shell
$ cd documents/
$ sphinx-quickstart
```

+ Write documents and put it under `source/` directory.

+ Compile documents to HTML, and the corresponding HTML file will be generated in `build/*.html`
```shell
$ cd documents/
$ make html
```

+ Check HTML correctness
```shell
$ cd documents/
$ python3 -m http.server
```
Then open the IP address with a web browser. For example, `http://0.0.0.0:8000/build/`. If the website is not updated, please remove the contents under `build/` directory and rebuild it.

+ render HTML on gitlab
Refer to [Gitlab Pages](https://docs.gitlab.com/ee/user/project/pages/).
