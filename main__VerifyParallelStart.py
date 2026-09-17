import os
import glob
import numpy as np
import pyvista as pv
import filecmp


def load_vtk_fields(filepath):
    """
    Reads a VTK file (legacy, XML, or Parallel) using PyVista
    and extracts the fq_XX fields as NumPy arrays.
    """
    try:
        mesh = pv.read(filepath)
    except Exception as e:
        print(f"    [!] Failed to read {filepath}: {e}")
        return None
    
    nx = mesh.dimensions[0] - 1
    ny = mesh.dimensions[1] - 1
    nz = mesh.dimensions[2] - 1
    print(f" Reading {filepath} ({nz},{ny},{nx})")
    fields = {}

    for q in range(19):
        field_name = f"fq_{q:02d}"

        if field_name in mesh.cell_data:
            fields[field_name] = np.array(mesh.cell_data[field_name]).reshape(nz, ny, nx)

        elif field_name in mesh.point_data:
            fields[field_name] = np.array(mesh.point_data[field_name]).reshape(nz, ny, nx)

        else:
            print(
                f"    [!] Warning: Field '{field_name}' "
                f"not found in {filepath}"
            )

    return fields


def detailed_vtk_analysis(base_filepath, other_filepath):

    print(
        "    [*] Comparing actual VTK files and decoded arrays..."
    )

    # ==============================================================
    # 1. ACTUAL FILE COMPARISON
    # ==============================================================
    files_equal = filecmp.cmp(
        base_filepath,
        other_filepath,
        shallow=False
    )

    if not files_equal: print("    [FAIL] VTK files: different bytes")

    # ==============================================================
    # 2. LOAD ARRAYS
    # ==============================================================

    base_fields = load_vtk_fields(base_filepath)
    other_fields = load_vtk_fields(other_filepath)

    if not base_fields or not other_fields:
        print("    [FAIL] Could not load fields from one or both files.")
        return False

    arrays_equal = True
    current_method_equal = True

    # ==============================================================
    # 3. ARRAY COMPARISONS
    # ==============================================================

    for q in range(19):

        field_name = f"fq_{q:02d}"

        if field_name not in base_fields:
            print(f"    [FAIL] {field_name} missing from base file")
            arrays_equal = False
            current_method_equal = False
            continue

        if field_name not in other_fields:
            print(f"    [FAIL] {field_name} missing from other file")
            arrays_equal = False
            current_method_equal = False
            continue

        base_arr = base_fields[field_name]
        other_arr = other_fields[field_name]

        # ----------------------------------------------------------
        # Shape
        # ----------------------------------------------------------
        if base_arr.shape != other_arr.shape:
            print(
                f"    [FAIL] {field_name}: shape mismatch "
                f"{base_arr.shape} vs {other_arr.shape}"
            )
            arrays_equal            = False
            current_method_equal    = False
            continue

        # ----------------------------------------------------------
        # Method 1: np.array_equal()
        # ----------------------------------------------------------
        exact_equal = np.array_equal(
            base_arr,
            other_arr
        )

        # ----------------------------------------------------------
        # Method 2: Current != method
        # ----------------------------------------------------------
        mismatch_mask = base_arr != other_arr
        num_mismatches = np.count_nonzero(mismatch_mask)
        current_equal = (num_mismatches == 0)

        # ----------------------------------------------------------
        # Report
        # ----------------------------------------------------------
        if not exact_equal:     arrays_equal = False
        if not current_equal:   current_method_equal = False

    # ==============================================================
    # 4. FINAL RESULT
    # ==============================================================
    print()
    print("    ===============================================")
    print("    COMPARISON SUMMARY")
    print("    ===============================================")
    print(
        f"    VTK files byte-for-byte identical : "
        f"{files_equal}"
    )
    print(
        f"    np.array_equal()                   : "
        f"{arrays_equal}"
    )
    print(
        f"    != element comparison              : "
        f"{current_method_equal}"
    )

    print("    ===============================================")

    return (
        files_equal
        and arrays_equal
        and current_method_equal
    )


# =================================================================
# DEFINE YOUR FOLDERS HERE
# =================================================================

folders = [
    
    "../DEBUG_PARALLEL_INIT/1_1_1_gpu/vis000/",
    "../DEBUG_PARALLEL_INIT/1_1_2_gpu/vis000/",
    "../DEBUG_PARALLEL_INIT/1_2_1_gpu/vis000/",
    "../DEBUG_PARALLEL_INIT/2_1_1_gpu/vis000/",
    "../DEBUG_PARALLEL_INIT/2_2_2_gpu/vis000/",

    "../DEBUG_PARALLEL_INIT/1_1_1/vis000/",
    "../DEBUG_PARALLEL_INIT/1_1_2/vis000/",
    "../DEBUG_PARALLEL_INIT/1_2_1/vis000/",
    "../DEBUG_PARALLEL_INIT/2_1_1/vis000/",
    "../DEBUG_PARALLEL_INIT/2_2_2/vis000/",
]


base_folder = folders[0]

search_pattern_pvti = os.path.join(
    base_folder,
    "summary.pvti"
)

base_files = glob.glob(search_pattern_pvti)

all_match = True

print(
    f"Found {len(base_files)} VTK debug file(s) "
    f"in base folder '{base_folder}'.\n"
)

for base_filepath in sorted(base_files):

    filename = os.path.basename(base_filepath)

    print(f"--- Checking {filename} ---")

    for other_folder in folders[1:]:

        other_filepath = os.path.join(
            other_folder,
            filename
        )

        if not os.path.exists(other_filepath):

            print(
                f"  [FAIL] Missing in '{other_folder}'"
            )

            all_match = False
            continue

        print(
            f"\n  Comparing '{base_folder}' "
            f"vs '{other_folder}'..."
        )

        result = detailed_vtk_analysis(
            base_filepath,
            other_filepath
        )

