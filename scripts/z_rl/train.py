"""Train a project task with LocoLab's Z-RL entry point."""

from locolab.entrypoints.z_rl.train import main

if __name__ == "__main__":
    # Task-specific help resolves presets before the shared entry point loads tasks.
    import locolab.tasks  # noqa: F401

    main()
