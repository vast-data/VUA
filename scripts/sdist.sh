#!/bin/bash

# This standalone script creates a soruce tarball.

set -eu
set -o pipefail
set +o posix
shopt -s inherit_errexit

main() {
    if ! git rev-parse --show-toplevel >/dev/null 2>/dev/null ; then
        echo "Not a git repository (or Git not installed)"
        exit -1
    fi

    if [[ -n "$(git status --porcelain)" ]]; then
        echo "'git status' needs to be clean for build-src.sh"
        exit -1
    fi

    set -e

    RDIR=$(mktemp -d -t buildsh-XXXXXXXXXX)
    NAME=uva-$(git describe)

    echo
    echo Creating ${NAME}.tar.gz
    echo

    TDIR=${RDIR}/${NAME}

    rm -rf src-dist
    mkdir src-dist

    mkdir ${TDIR}
    git archive --format=tar HEAD | tar -xf - -C ${TDIR}
    git describe > ${TDIR}/.git-describe
    DF=$(realpath src-dist)/${NAME}.tar.gz
    tar -cf - -C ${RDIR} ${NAME} | gzip -c > ${DF}

    ls -l src-dist

    rm -rf ${TDIR}
    rmdir ${RDIR}
}

main
