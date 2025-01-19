help:
	bash run.sh help

clean:
	bash run.sh clean

install-package:
	bash run.sh install:package

install-dev:
	bash run.sh install:dev

install-docs:
	bash run.sh install:docs

model:
	bash run.sh model

pre-commit-install:
	bash run.sh pre-commit:install

pre-commit-run:
	bash run.sh pre-commit:run

pre-commit-update:
	bash run.sh pre-commit:update
