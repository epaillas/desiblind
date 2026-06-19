"""Compatibility wrapper for the data-vector blinding parameter generator.

Prefer running ``scripts/data_vector_blinding.py`` or the
``desiblind-data-vector-parameters`` console script.
"""

from desiblind.data_vector_parameters import main


if __name__ == "__main__":
    main()
