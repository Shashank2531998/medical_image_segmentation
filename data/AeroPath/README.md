This README file was generated on 03-11-2023 by David Bouget.
Last updated: 03-11-2023.

--------------------
GENERAL INFORMATION
--------------------
1. Title of Dataset: AeroPath
2. Publication and DOI: "AeroPath: An airway segmentation benchmark dataset with challenging pathology". https://arxiv.org/abs/2311.01138
3. Contact Information
     Name: Erlend F. Hofstad
     Institution: Medical Technology Department, SINTEF Digital, Trondheim
     Email: erlend.hofstad@sintef.no
     Website: https://www.sintef.no/en/all-employees/employee/erlend.hofstad/


4. Contributors: Karen-Helene Støverud, Haakon Olav Leira, Erlend F. Hofstad, Andre Pedersen, David Bouget, and Thomas Langø.
5. Kind of data: computed tomography angiography (CTA) scans and binary annotation masks, all stored as NifTI files (*.nii.gz).

6. Date of data collection/generation: .
7. Geographic location: Trondheim, Norway.
8. Funding sources: the Ministry of Health and Care Services of Norway through the Norwegian National Research Center for Minimally Invasive and Image-Guided Diagnostics and Therapy (MiDT) at St. Olavs hospital, Trondheim University Hospital, Trondheim, Norway. The research leading to these results has in addition received funding from the Norwegian Financial Mechanism 2014-2021 under the project RO- NO2019-0138, 19/2020 “Improving Cancer Diagnostics in Flexible Endoscopy using Artificial Intelligence and Medical Robotics” IDEAR, Contract No. 19/2020.


9. Description of dataset: 
General description and ethics approvals: The dataset contains 27 computed tomography angiography
(CTA) scans, acquired using the Thorax Lung protocol at St. Olavs hospital (Trondheim, Norway). The included patients (nine women), aged 52 to 84 (median 70), were all undergoing diagnostic tests for lung cancer and had a wide range of pathologies including malignant tumors, sarcoidosis, and emphysema.


---------------------------
SHARING/ACCESS INFORMATION
---------------------------

1. Licenses/Restrictions:CC-BY 4.0 (See license.md).
2. Recommended citation: See citation recommended at https://github.com/raidionics/AeroPath.


---------------------
DATA & FILE OVERVIEW
---------------------
1. File List: 
README.md
license.md
 └── 1/
        └── 1_CT_HR.nii.gz
        └── 1_CT_HR_label_airways.nii.gz
        └── 1_CT_HR_label_lungs.nii.gz
 .
 .
 .
 └── 27/
        └── 27_CT_HR.nii.gz
        └── 27_CT_HR_label_airways.nii.gz
        └── 27_CT_HR_label_lungs.nii.gz

---------------------------
METHODOLOGICAL INFORMATION
---------------------------
1. Description of sources and methods used for collection/generation of data:
Dataset statistics
Overall, the CT scan dimensions are covering [487 : 512] × [441 : 512] × [241 : 829] voxels, and the trans-axial voxel size ranges [0.68 : 0.76] × [0.68 : 0.75] mm2 with a reconstructed slice thickness of [0.5 : 1.25] mm.

Annotation procedures
The annotation process for generating the ground truth was performed in three steps. First, the largest components (i.e., trachea and the first branches) were extracted based on a region growing, or a grow-cut method. Due to leakage, the region growing method did not yield satisfactory results in all cases. Therefore, for certain cases, the grow-cut method in 3D Slicer was used instead. In the second step, BronchiNet was employed to segment the smaller peripheral airways. In the third and final step, the segmentations were refined manually. Bronchial fragments and missed segments were connected, before false positives and fragments that could not be connected based on visual inspection were removed. All manual corrections were performed using the default segment editor in 3D Slicer. The manual correction was performed by a trained engineer, supervised by a pulmonologist. Finally, all annotations were verified on a case-by-case basis by a pulmonologist. The final annotations from the AeroPath segmentation included on average 128 ± 56 branches per CT scan.
