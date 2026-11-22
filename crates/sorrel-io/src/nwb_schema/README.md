# Cached NWB namespaces

These are the `/specifications` JSON datasets emitted by `NWBHDF5IO.write`
in PyNWB 4.2.0 with HDMF 6.2.0, reformatted as compact JSON. The native
writer uses them unmodified to describe NWB core 2.11.0, HDMF Common
1.10.0 and HDMF Experimental 0.6.0. To regenerate, write a minimal
`NWBFile` with `cache_spec=True` and extract the scalar JSON datasets by
namespace, version and dataset name.

Upstream: [nwb-schema](https://github.com/NeurodataWithoutBorders/nwb-schema)
([license](LICENSE-nwb.txt)) and
[hdmf-common-schema](https://github.com/hdmf-dev/hdmf-common-schema)
([license](LICENSE-hdmf.txt)).
