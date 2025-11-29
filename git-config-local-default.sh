#!/bin/bash

CHARSET="UTF-8"

gitConfigLocal () {
	local config="$1"
	local commandGitConfig="git config --local $config"

	echo "$commandGitConfig"
	$commandGitConfig
}

gitConfigLocal "color.ui true"

gitConfigLocal "i18n.commitencoding $CHARSET"

gitConfigLocal "i18n.logoutputencoding $CHARSET"

gitConfigLocal "gui.encoding $CHARSET"

gitConfigLocal "branch.autosetuprebase always"

gitConfigLocal "pull.rebase true"

gitConfigLocal "core.safecrlf warn"

gitConfigLocal "core.quotepath false"

gitConfigLocal "core.preloadindex true"

gitConfigLocal "core.filemode false"

gitConfigLocal "core.fscache true"

gitConfigLocal "gc.auto 256"

operatingSystem="$(uname -o 2>&1)"

if [ $? -eq 0 ]; then
	operatingSystem="${operatingSystem,,}"

	if [ "$operatingSystem" = "msys" ]; then
		gitConfigLocal "core.longpaths true"

		gitConfigLocal "core.ignorecase true"
	fi
else
	echo "$operatingSystem"
fi