"""Script to generate a html table of contributors, with names and avatars.

The list is generated from aeon's teams on GitHub, plus a small number
of hard-coded contributors.

The table should be updated for each new inclusion in the teams.
Generating the table requires admin rights.

This script is based on the script used by the scikit-learn project from 03/07/2023:
https://github.com/scikit-learn/scikit-learn/blob/main/build_tools/
generate_authors_table.py
"""

import sys
import time
from os import path
from pathlib import Path

import requests

LOGO_URL = "https://avatars2.githubusercontent.com/u/78909809"
REPO_FOLDER = Path(path.abspath(__file__)).parent.parent


def get(url, auth):
    """Get a URL, retrying if the rate limit is exceeded."""
    for sleep_time in [10, 30, 0]:
        reply = requests.get(url, auth=auth)
        api_limit = (
            "message" in reply.json()
            and "API rate limit exceeded" in reply.json()["message"]
        )
        if not api_limit:
            break
        print("API rate limit exceeded, waiting..")  # noqa: T201
        time.sleep(sleep_time)

    reply.raise_for_status()
    return reply


def get_contributors(auth):
    """Get the list of contributor profiles. Require admin rights."""
    cocw = []
    cocw_slug = "aeon-code-of-conduct-workgroup"
    cw = []
    cw_slug = "aeon-communications-workgroup"
    cd = []
    cd_slug = "aeon-core-developers"
    fw = []
    fw_slug = "aeon-finance-workgroup"
    iw = []
    iw_slug = "aeon-infrastructure-workgroup"
    rmw = []
    rmw_slug = "aeon-release-management-workgroup"

    entry_point = "https://api.github.com/orgs/aeon-toolkit/"

    for team_slug, lst in zip(
        (cocw_slug, cw_slug, cd_slug, fw_slug, iw_slug, rmw_slug),
        (cocw, cw, cd, fw, iw, rmw),
    ):
        for page in range(5):  # 5 pages, 30 per page
            reply = get(f"{entry_point}teams/{team_slug}/members?page={page}", auth)
            lst.extend(reply.json())

    # keep only the logins
    cocw = set(c["login"] for c in cocw)
    cw = set(c["login"] for c in cw)
    cd = set(c["login"] for c in cd)
    fw = set(c["login"] for c in fw)
    iw = set(c["login"] for c in iw)
    rmw = set(c["login"] for c in rmw)

    # add missing contributors with GitHub accounts
    cocw |= {"KatieBuc"}

    # get profiles from GitHub
    cocw = [get_profile(login, auth) for login in cocw]
    cw = [get_profile(login, auth) for login in cw]
    cd = [get_profile(login, auth) for login in cd]
    fw = [get_profile(login, auth) for login in fw]
    iw = [get_profile(login, auth) for login in iw]
    rmw = [get_profile(login, auth) for login in rmw]

    # sort by last name
    cocw = sorted(cocw, key=key)
    cw = sorted(cw, key=key)
    cd = sorted(cd, key=key)
    fw = sorted(fw, key=key)
    iw = sorted(iw, key=key)
    rmw = sorted(rmw, key=key)

    return (
        cocw,
        cw,
        cd,
        fw,
        iw,
        rmw,
    )


def get_profile(login, auth):
    """Get the GitHub profile from login."""
    print("get profile for %s" % (login,))  # noqa: T201
    try:
        profile = get("https://api.github.com/users/%s" % login, auth).json()
    except requests.exceptions.HTTPError:
        return dict(name=login, avatar_url=LOGO_URL, html_url="")

    if profile["name"] is None:
        profile["name"] = profile["login"]

    # fix missing names
    missing_names = {
        "KatieBuc": "Katie Buchhorn",
    }
    if profile["name"] in missing_names:
        profile["name"] = missing_names[profile["name"]]

    return profile


def key(profile):
    """Get a sorting key based on the lower case last name, then first name."""
    components = profile["name"].lower().split(" ")
    return " ".join([components[-1]] + components[:-1])


def generate_table(contributors):
    """Generate the html table from the list of contributors."""
    lines = [
        ".. raw :: html\n",
        "    <!-- Generated by generate_developer_table.py -->",
        '    <div class="aeon-teams-container">',
    ]
    for contributor in contributors:
        lines.append("    <div>")
        lines.append(
            "    <a href='%s'><img src='%s' class='avatar' /></a> <br />"
            % (contributor["html_url"], contributor["avatar_url"])
        )
        lines.append(
            "    <p><a href='%s'>%s</a></p>"
            % (contributor["html_url"], contributor["name"])
        )
        lines.append("    </div>")
    lines.append("    </div>")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print("access token:", file=sys.stderr)  # noqa: T201
    token = input()
    auth = ("user", token)

    (
        cocw,
        cw,
        cd,
        fw,
        iw,
        rmw,
    ) = get_contributors(auth)

    with open(
        REPO_FOLDER / "docs" / "about" / "code_of_conduct_workgroup.rst",
        "w+",
        encoding="utf-8",
    ) as rst_file:
        rst_file.write(generate_table(cocw))

    with open(
        REPO_FOLDER / "docs" / "about" / "communications_workgroup.rst",
        "w+",
        encoding="utf-8",
    ) as rst_file:
        rst_file.write(generate_table(cw))

    with open(
        REPO_FOLDER / "docs" / "about" / "core_developers.rst", "w+", encoding="utf-8"
    ) as rst_file:
        rst_file.write(generate_table(cd))

    with open(
        REPO_FOLDER / "docs" / "about" / "finance_workgroup.rst", "w+", encoding="utf-8"
    ) as rst_file:
        rst_file.write(generate_table(fw))

    with open(
        REPO_FOLDER / "docs" / "about" / "infrastructure_workgroup.rst",
        "w+",
        encoding="utf-8",
    ) as rst_file:
        rst_file.write(generate_table(iw))

    with open(
        REPO_FOLDER / "docs" / "about" / "release_management_workgroup.rst",
        "w+",
        encoding="utf-8",
    ) as rst_file:
        rst_file.write(generate_table(rmw))
