# Data

The DNS database is not included (about 7 GB). Download the 29-case
parameterized periodic-hill database (Re = 5600) from the NASA Langley
Turbulence Modeling Resource:

https://turbmodels.larc.nasa.gov/Other_DNS_Data/parameterized_periodic_hills.html

or from the original repository, https://github.com/xiaoh/para-database-for-PIML
(folder `pehill-29-cases-DNS`).

Place the case folders here so the layout is

```
data/pehill-29-cases-DNS/
    alph05-4071-2024/
        mean_files.dat
        rms_files1.dat
        rms_files2.dat
        diss_vort_files.dat
        yp.dat
    alph05-4071-3036/
    ...
```

Only the `.dat` files are read; the `.vtr` files can be ignored. If the data
lives somewhere else, point the code at it with

```
export PEHILL_DATA=/path/to/pehill-29-cases-DNS
```

Check that everything parses with `python src/data_loader.py`.
