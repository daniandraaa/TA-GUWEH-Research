"""Same V4 CLI arguments, with audited LeadTime backend compatibility."""
from .runtime import install
from research_v4.__main__ import main


if __name__ == '__main__':
    install()
    main()
