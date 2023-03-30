# Contributing guidelines

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md).
- Read [Code of Conduct](CODE_OF_CONDUCT.md).
- Ensure you have signed the [Contributor License Agreement (CLA)](https://cla-assistant.io/alibaba/DeepRec).
- Check if my changes are consistent with the [guidelines](https://github.com/alibaba/DeepRec/blob/main/CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution).
- Changes are consistent with the [Coding Style](https://github.com/alibaba/DeepRec/blob/main/CONTRIBUTING.md#c-coding-style).
- Run [Unit Tests](https://github.com/alibaba/DeepRec/blob/main/CONTRIBUTING.md#running-unit-tests).

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement (CLA).

  * If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](https://cla-assistant.io/alibaba/DeepRec).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](https://cla-assistant.io/alibaba/DeepRec).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

### Contributing code

If you have improvements to DeepRec, send us your pull requests! For those
just getting started, Github has a
[how to](https://help.github.com/articles/using-pull-requests/).

DeepRec team members will be assigned to review your pull requests. Once the
pull requests are approved and pass continuous integration checks, a DeepRec
team member will apply `ready to pull` label to your change. This means we are
working on getting your pull request submitted to our internal repository. After
the change has been submitted internally, your pull request will be merged
automatically on GitHub.

If you want to contribute, start working through the DeepRec codebase,
navigate to the
[Github "issues" tab](https://github.com/alibaba/DeepRec/issues) and start
looking through interesting issues. If you are not sure of where to start, then
start by trying one of the smaller/easier issues here i.e.
[issues with the "good first issue" label](https://github.com/alibaba/DeepRec/labels/good%20first%20issue)
and then take a look at the
[issues with the "contributions welcome" label](https://github.com/alibaba/DeepRec/labels/stat%3Acontributions%20welcome).
These are issues that we believe are particularly well suited for outside
contributions, often because we probably won't get to them right now. If you
decide to start on an issue, leave a comment so that other people know that
you're working on it. If you want to help out, but not alone, use the issue
comment thread to coordinate.

### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/alibaba/DeepRec/pulls),
make sure your changes are consistent with the guidelines and follow the
DeepRec coding style.

#### General guidelines and philosophy for contribution

*   Include unit tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates insufficient test coverage.
*   Keep API compatibility in mind when you change code in core DeepRec,
    e.g., code in
    [core](https://github.com/alibaba/DeepRec/tree/main/tensorflow/core)
    and
    [python](https://github.com/alibaba/DeepRec/tree/main/tensorflow/python).
    DeepRec has reached version 1 and hence cannot make
    non-backward-compatible API changes without a major release. Reviewers of
    your pull request will comment on any API compatibility issues.
*   When you contribute a new feature to DeepRec, the maintenance burden is
    (by default) transferred to the DeepRec team. This means that the benefit
    of the contribution must be compared against the cost of maintaining the
    feature.
*   Full new features (e.g., a new op implementing a cutting-edge algorithm)
    typically will live in
    [tensorflow/addons](https://github.com/tensorflow/addons) to get some
    airtime before a decision is made regarding whether they are to be migrated
    to the core.

#### License

Include a license at the top of new files.

* [C/C++ license example](https://github.com/alibaba/DeepRec/blob/main/tensorflow/core/framework/op.cc#L1)
* [Python license example](https://github.com/alibaba/DeepRec/blob/main/tensorflow/python/ops/nn.py#L1)
* [Java license example](https://github.com/alibaba/DeepRec/blob/main/tensorflow/java/src/main/java/org/tensorflow/Graph.java#L1)
* [Go license example](https://github.com/alibaba/DeepRec/blob/main/tensorflow/go/operation.go#L1)
* [Bash license example](https://github.com/alibaba/DeepRec/blob/main/tensorflow/tools/ci_build/ci_sanity.sh#L2)
* [HTML license example](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/components/tf_backend/tf-backend.html#L2)
* [JavaScript/TypeScript license example](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/components/tf_backend/backend.ts#L1)

Bazel BUILD files also need to include a license section, e.g.,
[BUILD example](https://github.com/alibaba/DeepRec/blob/main/tensorflow/core/BUILD#L61).

#### C++ coding style

Changes to DeepRec C++ code should conform to
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

Use `clang-tidy` to check your C/C++ changes. To install `clang-tidy` on ubuntu:18.04, do:

```bash
apt-get install -y clang-tidy
```

You can check a C/C++ file by doing:


```bash
clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
diff <my_cc_file> /tmp/my_cc_file.cc
```

#### Python coding style

Changes to DeepRec Python code should conform to
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

Use `pylint` to check your Python changes. To install `pylint` and
retrieve DeepRec's custom style definition:

```bash
pip install pylint
wget -O /tmp/pylintrc https://raw.githubusercontent.com/alibaba/DeepRec/main/tensorflow/tools/ci_build/pylintrc
```

To check a file with `pylint`:

```bash
pylint --rcfile=/tmp/pylintrc myfile.py
```

#### Coding style for other languages

* [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html)
* [Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html)
* [Google Shell Style Guide](https://google.github.io/styleguide/shell.xml)
* [Google Objective-C Style Guide](https://google.github.io/styleguide/objcguide.html)

#### Git Commit Guidelines

Use meaningful commit message that described what you did. Format should be: [Component] <Subject> <Description>

Component: after using the Component mark, there needs to be a space with the following subject.

Subject: a brief description of the purpose of commit, must be in English, no more than 50 characters. The first letter needs to be capitalized and the ending is `.`

Description: Description is a further detailed description of the commit, such as bugfix, which is the scene caused by the bug. If it is performance optimization, it is performance data. Use the description as a supplement to the subject.

Example:

```
[Runtime] Add blacklist and whitelist to JitCugraph. (#578)

1. Refine the auto-clustering policy by adding blacklist and whitelist environment setup.
2. Add documents of using JitCugraph.
```

#### Running unit tests

Using Docker and DeepRec's CI scripts.

```bash
# Install Docker first, then this will build and run cpu tests
cibuild/cpu-ut/cpu-core-ut.sh
```
Also you can directly use bazel to run the tests, like:

```bash
./configure
bazel test //tensorflow/python/...
```

See
[DeepRec Builds](https://github.com/alibaba/DeepRec/tree/main/cibuild)
for details.
