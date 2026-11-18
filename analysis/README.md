# Analysis code

Keep reusable analysis scripts here. Raw input and generated figures/reports
belong in the outer workspace's local_results/<incoming|analysis>/<run-id>.

The current late-H5 report and its reproducible script remain at:
../../colab/detonation_tube/h5_salvage_analysis_20261117/

They have not been bulk-moved because the historical script uses relative paths
and archived input provenance. A future reusable CLI must accept explicit input
and output paths instead of assuming a particular Colab folder layout.
