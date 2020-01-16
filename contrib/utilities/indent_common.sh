#!/bin/bash

#
# This file contains a number of common functions used all indent scripts
#

export DIFFUSION_CLANG_FORMAT="clang-format-6.0"


#
# This function checks that we are in the root directory and that
# clang-format is available. It is called by both indent and indent-all
# to ensure that the rest of the indentation pipeline makes sense.
#
checks() {
  if test ! -d source -o ! -d include -o ! -d doc ; then
    echo "*** This script must be run from the top-level directory."
    exit 1
  fi

  if ! [ -x "$(command -v ${DIFFUSION_CLANG_FORMAT})" ]; then
    echo "***"
    echo "***   No clang-format-6.0 program found. Install form your package manager."
    echo "***"
    exit 1
  fi

  # Make sure to have the right version. We know that clang-6.0.0
  # and clang-6.0.1 work. Hence, test for clang-6.0.
  CLANG_FORMAT_VERSION="$(${DIFFUSION_CLANG_FORMAT} --version)"
  CLANG_FORMAT_MAJOR_VERSION=$(echo "${CLANG_FORMAT_VERSION}" | sed 's/^[^0-9]*\([0-9]*\).*$/\1/g')
  CLANG_FORMAT_MINOR_VERSION=$(echo "${CLANG_FORMAT_VERSION}" | sed 's/^[^0-9]*[0-9]*\.\([0-9]*\).*$/\1/g')
  if [ "${CLANG_FORMAT_MAJOR_VERSION}" -ne 6 ] || [ "${CLANG_FORMAT_MINOR_VERSION}" -ne 0 ]; then
    echo "***"
    echo "***   This indent script requires clang-format version 6.0. Install form your package manager."
    echo "***"
    exit 1
  fi


  # check formatting of usernames and email addresses, examples that will be detected:
  # not-using-a-name <a@b.com>
  # John Doe <doe@macbook.local>
  # Jane Doe <a@nodomain>
  #
  # first user names:
  git log --since "2020-01-01" --format="%aN" | sort -u | while read name ; do
      words=($name)
      if [ "${#words[@]}" -lt "2" ]; then
	  echo "invalid author '$name' without firstname and lastname"
	  echo ""
	  echo "hint:"
	  echo "   git config --global user.name 'FIRST_NAME LAST_NAME'"
	  exit 2
      fi
  done || exit 2

  # now emails:
  git log --since "2019-01-01" --format="%aE" | sort -u | while read email ; do
      words=($name)
      if ! echo "$email" | grep -q "\."; then
	  echo "invalid email '$email'"
          echo ""
          echo "hint:"
          echo "   git config --global user.email 'MY_NAME@example.com'"
	  exit 3
      fi
      if ! echo "$email" | grep -q -v -e "\.local$"; then
	  echo "invalid email '$email'"
          echo ""
          echo "hint:"
          echo "   git config --global user.email 'MY_NAME@example.com'"
	  exit 3
      fi
  done || exit 3
}


#
# Mac OSX's mktemp doesn't know the --tmpdir option without argument. So,
# let's do all of that by hand:
#
export TMPDIR="${TMPDIR:-/tmp}"

export REPORT_ONLY="false"


#
# If REPORT_ONLY is set to "true", this function reports a formatting issue
# if file "${1}" and tmpfile "${2}" don't match (using the error message
# "${3}"), or, if set to "false" silently replaces file "${1}" with "${2}".
#
fix_or_report()
{
  file="${1}"
  tmpfile="${2}"
  message="${3}"

  if ! diff -q "${file}" "${tmpfile}" >/dev/null; then
    if ${REPORT_ONLY}; then
      echo "    ${file}  -  ${message}"
    else
      mv "${tmpfile}" "${file}"
    fi
  fi
}
export -f fix_or_report

#
# In order to format .cc and .h files we have to make sure that we override
# the source/header file only if the actual contents changed.
# Unfortunately, clang-format isn't exactly helpful there. Thus, use a
# temporary file and diff as a workaround.
#
format_file()
{
  file="${1}"
  tmpfile="$(mktemp "${TMPDIR}/$(basename "$1").tmp.XXXXXXXX")"
  arg="-style=file"

  "${DIFFUSION_CLANG_FORMAT}" "${arg}" "${file}" > "${tmpfile}"
  fix_or_report "${file}" "${tmpfile}" "file indented incorrectly"
  rm -f "${tmpfile}"
}
export -f format_file


#
# Convert DOS formatted files to unix file format by stripping out
# carriage returns (15=0x0D):
#
dos_to_unix()
{
  file="${1}"
  tmpfile="$(mktemp "${TMPDIR}/$(basename "$1").tmp.XXXXXXXX")"

  tr -d '\015' <"${file}" >"${tmpfile}"

  fix_or_report "${file}" "${tmpfile}" "file has non-unix line-ending '\\r\\n'"
  rm -f "${tmpfile}" "${tmpfile}"
}
export -f dos_to_unix


#
# Fix permissions
#
fix_permissions()
{
  file="${1}"

  case "${OSTYPE}" in
    darwin*)
      PERMISSIONS="$(stat -f '%a' ${file})"
      ;;
    *)
      PERMISSIONS="$(stat -c '%a' ${file})"
      ;;
  esac

  if [ "${PERMISSIONS}" != "644" ]; then
    if ${REPORT_ONLY}; then
      echo "    ${file}  -  file has incorrect permissions"
    else
      chmod 644 "${file}"
    fi
  fi
}
export -f fix_permissions


#
# Collect all files found in a list of directories "${1}$" matching a
# regular expression "${2}$", and process them with a command "${3}" on 10
# threads in parallel.
#
# The command line is a bit complicated, so let's discuss the more
# complicated arguments:
# - For 'find', -print0 makes sure that file names are separated by \0
#   characters, as opposed to the default \n. That's because, surprisingly,
#   \n is a valid character in a file name, whereas \0 is not -- so it
#   serves as a good candidate to separate individual file names.
# - For 'xargs', -0 does the opposite: it separates filenames that are
#   delimited by \0
# - the options "-n 1 -P 10" make sure that the following script will be
#   called exactly with one file name as argument at a time, but we allow
#   execution for up to 10 times in parallel
#
process()
{
  case "${OSTYPE}" in
    darwin*)
      find -E ${1} -regex "${2}" -print0 |
        xargs -0 -n 1 -P 10 -I {} bash -c "${3} {}"
      ;;
    *)
      find ${1} -regextype egrep -regex "${2}" -print0 |
        xargs -0 -n 1 -P 10 -I {} bash -c "${3} {}"
      ;;
  esac
}


#
# Variant of above function that only processes files that have changed
# since the last merge commit to master. For this, we collect all files
# that
#  - are new
#  - have changed since the last merge commit to master
#
process_changed()
{
  LAST_MERGE_COMMIT="$(git log --format="%H" --merges --max-count=1 master)"
  COMMON_ANCESTOR_WITH_MASTER="$(git merge-base ${LAST_MERGE_COMMIT} HEAD)"

  case "${OSTYPE}" in
    darwin*)
      XARGS="xargs -E"
      ;;
    *)
      XARGS="xargs --no-run-if-empty -d"
      ;;
  esac

  ( git ls-files --others --exclude-standard -- ${1};
    git diff --name-only --diff-filter=d $COMMON_ANCESTOR_WITH_MASTER -- ${1} )|
      sort -u |
      grep -E "^${2}$" |
      ${XARGS} '\n' -n 1 -P 10 -I {} bash -c "${3} {}"
}