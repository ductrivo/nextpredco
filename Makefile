help:
	bash run.sh help

clean:
	bash run.sh clean

install-package:
	bash run.sh install:package

install-dev:
	bash run.sh install:dev
	bash run.sh install:package

install-docs:
	bash run.sh install:docs
	bash run.sh install:package

model:
	bash run.sh model

pre-commit-install:
	bash run.sh pre-commit:install

pre-commit-run:
	bash run.sh pre-commit:run

pre-commit-update:
	bash run.sh pre-commit:update

test-model:
	bash run.sh test:model

test:
	bash run.sh test

test-cov:
	bash run.sh test:cov
