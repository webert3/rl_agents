import argparse
import rl_agents


def main(args):
    semver = rl_agents.__version__.split(".")

    with open("rl_agents/__init__.py", "w") as fp:
        new_semver = semver.copy()
        semver = ".".join(semver)

        if args.type.lower() == "major":
            new_semver[0] = str(int(new_semver[0]) + 1)
            new_semver = ".".join(new_semver)
            update_type = "MAJOR"
        elif args.type.lower() == "minor":
            new_semver[1] = str(int(new_semver[1]) + 1)
            new_semver = ".".join(new_semver)
            update_type = "MINOR"
        elif args.type.lower() == "patch":
            new_semver[2] = str(int(new_semver[2]) + 1)
            new_semver = ".".join(new_semver)
            update_type = "PATCH"

        print(f"Updating {update_type} version: {semver} -> {new_semver}")

        fp.write(f"__version__ = \"{new_semver}\"")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-t",
                        "--type",
                        type=str,
                        required=True,
                        choices=("major", "minor", "patch"),
                        help="Semantic version change type.")

    main(parser.parse_args())

