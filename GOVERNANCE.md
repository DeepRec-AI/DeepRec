*NOTE: This document is intended to provide an example governance structure for any LF AI and Data Foundation project to consider as a starting point. All projects hosted by LF AI and Data Foundation are not bound by these governance polices, but in absence of any prior governance structure should consider this as a recommended structure*

# Overview

This project aims to be governed in a transparent, accessible way for the benefit of the community. All participation in this project is open and not bound to corporate affilation. Participants are bound to the project's [Code of Conduct](./CODE_OF_CONDUCT.md).

# Project roles

## Contributor

The contributor role is the starting role for anyone participating in the project and wishing to contribute code.

# Process for becoming a contributor

* Review the [Contribution Guidelines](./CONTRIBUTING.md) to ensure your contribution is inline with the project's coding and styling guidelines.
* Submit your code as a PR with the CLA signoff
* Have your submission approved by the committer(s) and merged into the codebase.

## Committer

The committer role enables the contributor to commit code directly to the repository, but also comes with the responsibility of being a responsible leader in the community.

### Process for becoming a committer

* Show your experience with the codebase through contributions and engagement on the community channels.
* Provides advice and resources and shows leadership to guide or support the development of the project.
* Request to become a committer. To do this, create a new pull request that adds your name and details to the [Committers File](./COMMITTERS.md) file and request existing committers to approve.
* After the majority of committers approve you, merge in the PR. Be sure to tag the whomever is managing the GitHub permissions to update the committers team in GitHub.

### Committer responsibilities

* Monitor email aliases (if any).
* Monitor Slack (delayed response is perfectly acceptable).
* Triage GitHub issues and perform pull request reviews for other committers and the community.
* Make sure that ongoing PRs are moving forward at the right pace or closing them.
* In general continue to be willing to spend at least 25% of ones time working on the project (~1.25 business days per week).

### When does a committer lose committer status

If a committer is no longer interested or cannot perform the committer duties listed above, they
should volunteer to be moved to emeritus status. In extreme cases this can also occur by a vote of
the committers per the voting process below.

## Lead

The project committers will elect a lead ( and optionally a co-lead ) which will be the primary point of contact for the project and representative to the TAC upon becoming an Active stage project. The lead(s) will be responsible for the overall project health and direction, coordination of activities, and working with other projects and committees as needed for the continuted growth of the project.

# Release Process

DeepRec would be released every 2 months. Once there's any critical issue in release branch, after approved by leader or major committers approve, could release a patch.

Release process contains following steps:

* Submit a commit to update VERSION in setup.py, like: [[Release] Update DeepRec release version to 1.15.5+deeprec2302.](https://github.com/alibaba/DeepRec/commit/23252970336e92fafda3eac683c38ba08ca35e46)
* Create release branch and named like deeprec2302
* Create a release tag, named like r1.15-deeprec2302
* Build CPU/GPU wheels of DeepRec, please follow [Build DeepRec](https://deeprec.readthedocs.io/en/latest/DeepRec-Compile-And-Install.html)
* Build CPU/GPU wheels of Estimator, please follow [Build Estimator](https://deeprec.readthedocs.io/en/latest/Estimator-Compile-And-Install.html)
* Install the wheels of DeepRec and Estimator in base docker images which can be found in [Build DeepRec](https://deeprec.readthedocs.io/en/latest/DeepRec-Compile-And-Install.html).
* Run modelzoo and make sure all models are pass.
* Update README.md and list of user documents, please follow [[Docs] Update deeprec2212 release images in README.md and user docs.](https://github.com/alibaba/DeepRec/commit/6a47659034bd2fcd53685b8b07352a6defeabcab)
* Add release note, such as [Releases](https://github.com/alibaba/DeepRec/releases)


# Conflict resolution and voting

In general, we prefer that technical issues and committer membership are amicably worked out
between the persons involved. If a dispute cannot be decided independently, the committers can be
called in to decide an issue. If the committers themselves cannot decide an issue, the issue will
be resolved by voting. The voting process is a simple majority in which each committer receives one vote.

# Communication

This project, just like all of open source, is a global community. In addition to the [Code of Conduct](./CODE_OF_CONDUCT.md), this project will:

* Keep all communucation on open channels ( mailing list, dingtalk, wechat ).
* Be respectful of time and language differences between community members ( such as scheduling meetings, email/issue responsiveness, etc ).
* Ensure tools are able to be used by community members regardless of their region.

If you have concerns about communication challenges for this project, please contact the committers.

[Code of Conduct]: CODE_OF_CONDUCT.md
[Committers File]: COMMITTERS.md
[Contribution Guidelines]: CONTRIBUTING.md
