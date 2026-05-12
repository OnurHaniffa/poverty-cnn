# DHS Data Access Application

Project description text submitted to [dhsprogram.com](https://dhsprogram.com/data/new-user-registration.cfm) for access to household recode files and GPS shapefiles across 23 sub-Saharan African countries.

| | |
|---|---|
| Submitted | 2026-05-09 |
| Status | Awaiting approval |
| Countries requested | 23 (see below) |

---

## Project description (for DHS form)

The current research project builds upon the replication of Yeh et al. (2020, Nature Communications), "Using publicly available satellite imagery and deep learning to understand economic well-being in Africa". Specifically, the original paper trains a convolutional neural network on Landsat satellite imagery and nighttime lights to predict cluster-level asset wealth in 23 sub-Saharan African countries from DHS survey data.

This replication project modernizes the TensorFlow 1.15 deprecated code base to PyTorch 2.x and adds three innovations: (1) a country-level fairness audit of all 23 countries in the dataset, extending Aiken, Rolf, and Blumenstock (IJCAI 2023), which performed an audit on just 10 countries; (2) a novel fairness study that examines the distribution of uncertainty itself via Monte Carlo Dropout; (3) an examination of temporal drift in fairness measures between early (2009–2014) and late (2015–2017) DHS survey rounds.

I request household recode files and GPS shapefiles for 23 countries (Angola, Benin, Burkina Faso, Cameroon, Côte d'Ivoire, DRC, Ethiopia, Ghana, Guinea, Kenya, Lesotho, Malawi, Mali, Mozambique, Nigeria, Rwanda, Senegal, Sierra Leone, Tanzania, Togo, Uganda, Zambia, Zimbabwe).

The data will only be used for this academic research. Individual records will not be redistributed. The code will be open-sourced (MIT license), but the DHS data itself will not be included. Researcher: Onur Haniffa (Acibadem University), supervised by Dr. Seda Nilgün Dumlu.

---

## Justification for GPS / Geographic dataset access (for DHS form)

GPS coordinates are needed to pair each DHS cluster with its corresponding satellite imagery patch. Without GPS we can't extract the 6.72km Landsat tile centered on each cluster, which is the input to the convolutional neural network in this replication of Yeh et al. (2020, Nature Communications).

GPS data will only be used for spatial alignment between household surveys and satellite imagery. The cluster coordinates already include the standard DHS privacy displacement (up to 2km in urban clusters and 10km in rural clusters), and no attempt will be made to remove this or to identify any specific communities or households. All published results will only be reported at cluster-level or higher — never the raw GPS coordinates of a cluster.

This use of GPS together with satellite imagery follows the standard methodology of Jean et al. (2016, Science) and Yeh et al. (2020, Nature Communications), which has become the dominant approach for poverty mapping using publicly available remote sensing data.

---

## Countries requested (full list)

| # | Country | Most recent DHS round |
|---|---|---|
| 1 | Angola | 2015–16 |
| 2 | Benin | 2017–18 |
| 3 | Burkina Faso | 2010 |
| 4 | Cameroon | 2018 |
| 5 | Côte d'Ivoire | 2011–12 |
| 6 | DR Congo | 2013–14 |
| 7 | Ethiopia | 2016 |
| 8 | Ghana | 2014 |
| 9 | Guinea | 2018 |
| 10 | Kenya | 2014 |
| 11 | Lesotho | 2014 |
| 12 | Malawi | 2015–16 |
| 13 | Mali | 2018 |
| 14 | Mozambique | 2011 |
| 15 | Nigeria | 2018 |
| 16 | Rwanda | 2014–15 |
| 17 | Senegal | 2017 |
| 18 | Sierra Leone | 2013 |
| 19 | Tanzania | 2015–16 |
| 20 | Togo | 2013–14 |
| 21 | Uganda | 2016 |
| 22 | Zambia | 2018 |
| 23 | Zimbabwe | 2015 |

DHS rounds listed are approximate; verify against current DHS catalog when downloading. Some countries have newer rounds released since Yeh et al. 2020; for apples-to-apples replication we use the same rounds Yeh et al. used.

---

## Notes

- Format the project description as **Times New Roman 12pt** when pasting into the DHS web form (or whatever the form's default font is — most accept the description as plain text).
- Single application covers access to all 23 countries.
- Approval typically takes 1–3 business days. The DHS team may request clarifications by email.
- After approval, individual datasets (HR + GE files per country) must be downloaded country-by-country from the DHS website.
