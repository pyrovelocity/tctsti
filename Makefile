.DEFAULT_GOAL := help

ENV_PREFIX ?= ./
ENV_FILE := $(wildcard $(ENV_PREFIX)/.env)

ifeq ($(strip $(ENV_FILE)),)
$(info $(ENV_PREFIX)/.env file not found, skipping inclusion)
else
include $(ENV_PREFIX)/.env
export
endif

##@ Utility
help: ## Display this help. (Default)
# based on "https://gist.github.com/prwhite/8168133?permalink_comment_id=4260260#gistcomment-4260260"
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

##@ Utility
help_sort: ## Display alphabetized version of help.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

#-------------
# system / dev
#-------------

install_direnv: ## Install direnv to `/usr/local/bin`. Check script before execution: https://direnv.net/ .
	@which direnv > /dev/null || \
	(curl -sfL https://direnv.net/install.sh | bash && \
	sudo install -c -m 0755 direnv /usr/local/bin && \
	rm -f ./direnv)
	@echo "see https://direnv.net/docs/hook.html"

install_just: ## Install just. Check script before execution: https://just.systems/ .
	@which cargo > /dev/null || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)
	@cargo install just

install_poetry: ## Install poetry. Check script before execution: https://python-poetry.org/docs/#installation .
	@which poetry > /dev/null || (curl -sSL https://install.python-poetry.org | python3 -)

env_print: ## Print a subset of environment variables defined in ".env" file.
	env | grep "GITHUB\|GH_\|GCP_" | sort

ghsecrets: ## Update github secrets for GH_REPO from ".env" file.
	gh secret list --repo=$(GH_REPO)
	gh secret set GOOGLE_APPLICATION_CREDENTIALS_DATA --repo="$(GH_REPO)" --body='$(shell cat $(GCP_GACD_PATH))'
	gh secret set CODECOV_TOKEN --repo="$(GH_REPO)" --body="$(CODECOV_TOKEN)"
	gh secret set GCP_SERVICE_ACCOUNT --repo="$(GH_REPO)" --body="$(GCP_SERVICE_ACCOUNT)"
	gh secret set TEST_PYPI_TOKEN --repo="$(GH_REPO)" --body="$(TEST_PYPI_TOKEN)"
	gh secret list --repo=$(GH_REPO)

approve_prs: ## Approve github pull requests from bots: PR_ENTRIES="2-5 10 12-18"
	for entry in $(PR_ENTRIES); do \
		if [[ "$$entry" == *-* ]]; then \
			start=$${entry%-*}; \
			end=$${entry#*-}; \
			for pr in $$(seq $$start $$end); do \
				@gh pr review $$pr --approve; \
			done; \
		else \
			@gh pr review $$entry --approve; \
		fi; \
	done
