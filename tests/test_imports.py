def test_data_vector_compatibility_imports():
    from desiblind.blinding import TracerPowerSpectrumMultipolesBlinder as OldPowerBlinder
    from desiblind.data_vector import TracerPowerSpectrumMultipolesBlinder as NewPowerBlinder

    assert OldPowerBlinder is NewPowerBlinder
