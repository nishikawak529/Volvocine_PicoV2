function validation = test_global_joint_cp_rank_sweep_component_weights()
%TEST_GLOBAL_JOINT_CP_RANK_SWEEP_COMPONENT_WEIGHTS Noiseless regression test.
    validation = global_joint_cp_rank_sweep_component_weights([], 5, ...
        'SyntheticTestOnly', true, ...
        'NumStarts', 20, ...
        'MaxIter', 1000, ...
        'Tol', 1e-10, ...
        'RandomSeed', 0);
    expected = [9/14;13/14;1;1;1];
    assert(validation.ran);
    assert(validation.passed);
    assert(max(abs(validation.observed_explained_fraction-expected)) < 1e-9);
    assert(all(diff(validation.observed_explained_fraction) >= -1e-12));
    assert(all(validation.observed_relative_residual(3:5) < 1e-10));
    assert(validation.rank1_legacy_consistent);
    assert(validation.dimensions_valid);
    assert(validation.factors_valid);
    assert(validation.objectives_valid);
    assert(validation.reconstructions_valid);
end
