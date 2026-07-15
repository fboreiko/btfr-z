#!/usr/bin/env python3
"""
Script to combine five likelihood grid chunks into a single file, apply patches, and fill NaN values.
"""

import numpy as np
import os
import glob
import argparse

def print_nan_coordinates(grid):
    """
    Print the coordinates of NaN values in a 3D array.

    Parameters:
        grid (np.ndarray): The 3D array to check for NaN values.
    """
    # Find the indices where the grid is NaN
    nan_indices = np.argwhere(np.isnan(grid))
    
    if nan_indices.size > 0:
        print("Found NaNs at the following coordinates:")
        return nan_indices
    else:
        print("No NaNs found in the grid.")
        return None


def average_neighbors(arr, index):
    """
    Compute the average of the neighbors of the cell at 'index' in a 3D or 4D array.
    Only valid (non-NaN) neighbors within the bounds of the array are used.
    
    Parameters:
        arr (np.ndarray): 3D or 4D array.
        index (tuple): A tuple indicating the index of the cell.
    
    Returns:
        float: The average value of the neighbors. If no valid neighbor is found,
               returns np.nan.
    """
    neighbor_values = []
    dims = len(index)
    
    if dims not in [3, 4]:
        raise ValueError("This function only supports 3D or 4D arrays.")
    
    # Generate ranges for neighbors based on the number of dimensions
    ranges = [range(-1, 2) for _ in range(dims)]
    
    # Iterate over all possible neighbor offsets
    for offsets in np.ndindex(*[len(r) for r in ranges]):
        # Skip the center cell itself
        if all(offset == 0 for offset in offsets):
            continue
        
        # Compute neighbor indices
        neighbor_index = tuple(index[d] + offsets[d] for d in range(dims))
        
        # Check if the neighbor index is within bounds
        if all(0 <= neighbor_index[d] < arr.shape[d] for d in range(dims)):
            neighbor_val = arr[neighbor_index]
            # Only add non-NaN values
            if not np.isnan(neighbor_val):
                neighbor_values.append(neighbor_val)
    
    if neighbor_values:
        return np.mean(neighbor_values)
    else:
        return np.nan


def fill_nans_iteratively_for_all_mocks(grid_5d):
    """
    Iteratively fill NaN values for each of the 50 mock datasets individually.
    
    Parameters:
        grid_5d (np.ndarray): 5D grid with shape (50, 20, 20, 20, 20) where first dimension is mock datasets.
        
    Returns:
        np.ndarray: The grid with NaN values filled for each mock dataset.
    """
    print(f"Processing {grid_5d.shape[0]} mock datasets individually...")
    grid_filled = grid_5d.copy()
    
    total_nans_before = np.sum(np.isnan(grid_5d))
    print(f"Total NaN values across all mocks: {total_nans_before}")
    
    # Dictionary to store problematic indices for each mock
    problematic_mocks = {}
    
    for mock_idx in range(grid_5d.shape[0]):
        mock_grid = grid_5d[mock_idx]  # Shape: (20, 20, 20, 20)
        mock_nans = np.sum(np.isnan(mock_grid))
        
        if mock_nans > 0:
            print(f"\nProcessing mock dataset {mock_idx + 1}/{grid_5d.shape[0]} ({mock_nans} NaNs)")
            filled_mock = fill_nans_iteratively_single_grid(mock_grid)
            grid_filled[mock_idx] = filled_mock
            
            # Check if there are still NaNs after filling
            remaining_nans = np.sum(np.isnan(filled_mock))
            if remaining_nans > 0:
                nan_indices = np.argwhere(np.isnan(filled_mock))
                problematic_mocks[mock_idx] = nan_indices
                print(f"    Mock {mock_idx + 1} still has {remaining_nans} unfilled NaNs")
        else:
            print(f"Mock dataset {mock_idx + 1}/{grid_5d.shape[0]}: No NaNs found")
    
    total_nans_after = np.sum(np.isnan(grid_filled))
    print(f"\nTotal NaN values after processing all mocks: {total_nans_after}")
    print(f"Successfully filled {total_nans_before - total_nans_after} NaN values")
    
    # Save problematic indices to file if any exist
    if problematic_mocks:
        print(f"\nFound {len(problematic_mocks)} mock datasets with unfilled NaNs")
        output_file = "unfilled_nan_indices.npy"
        np.save(output_file, problematic_mocks)
        print(f"Saved problematic indices to {output_file}")
        
        # Print summary
        for mock_idx, indices in problematic_mocks.items():
            print(f"  Mock {mock_idx + 1}: {len(indices)} unfilled NaNs")
    
    return grid_filled


def fill_nans_iteratively_single_grid(grid):
    """
    Iteratively fill NaN values in a single 4D grid by averaging neighbors until no NaNs remain.
    
    Parameters:
        grid (np.ndarray): The 4D grid with potential NaN values.
        
    Returns:
        np.ndarray: The grid with NaN values filled.
    """
    grid_copy = grid.copy()
    iteration = 0
    
    while True:
        nan_indices = np.argwhere(np.isnan(grid_copy))
        
        if nan_indices.size == 0:
            if iteration > 0:
                print(f"    All NaNs filled after {iteration} iterations.")
            break
            
        if iteration == 0:
            print(f"    Starting with {len(nan_indices)} NaN values")
        
        # Keep track of values that were successfully filled this iteration
        filled_this_iteration = 0
        
        for nan_idx in nan_indices:
            index = tuple(nan_idx)
            averaged_value = average_neighbors(grid_copy, index)
            
            if not np.isnan(averaged_value):
                grid_copy[index] = averaged_value
                filled_this_iteration += 1
        
        if iteration % 10 == 0 and iteration > 0:  # Print progress every 10 iterations
            print(f"    Iteration {iteration + 1}: {len(nan_indices)} NaNs remaining")
        
        # If no values were filled this iteration, we're stuck
        if filled_this_iteration == 0:
            remaining_nans = len(nan_indices)
            print(f"    Warning: Could not fill {remaining_nans} remaining NaN values - no valid neighbors available")
            
            # Report the exact indices where NaNs remain
            print(f"    Unfilled NaN indices:")
            for i, idx in enumerate(nan_indices[:20]):  # Show first 20 to avoid too much output
                print(f"      {tuple(idx)}")
            
            # Check if these are at boundaries or isolated regions
            analyze_unfilled_nans(grid_copy, nan_indices)
            break
            
        iteration += 1
        
        # Safety check to prevent infinite loops
        if iteration > 1000:
            print("    Warning: Reached maximum iterations (1000). Stopping.")
            break
    
    return grid_copy


def analyze_unfilled_nans(grid, nan_indices):
    """
    Analyze the unfilled NaN values to understand why they couldn't be filled.
    
    Parameters:
        grid (np.ndarray): The grid with remaining NaN values.
        nan_indices (np.ndarray): Indices of the remaining NaN values.
    """
    print("    Analyzing unfilled NaN locations...")
    
    boundary_nans = 0
    isolated_nans = 0
    clustered_nans = 0
    
    for nan_idx in nan_indices:
        index = tuple(nan_idx)
        
        # Check if it's at a boundary
        is_boundary = any(idx == 0 or idx == grid.shape[dim] - 1 for dim, idx in enumerate(index))
        
        # Count valid neighbors
        neighbor_count = 0
        neighbor_nan_count = 0
        
        dims = len(index)
        ranges = [range(-1, 2) for _ in range(dims)]
        
        for offsets in np.ndindex(*[len(r) for r in ranges]):
            if all(offset == 0 for offset in offsets):
                continue
                
            neighbor_index = tuple(index[d] + offsets[d] for d in range(dims))
            
            if all(0 <= neighbor_index[d] < grid.shape[d] for d in range(dims)):
                neighbor_val = grid[neighbor_index]
                if np.isnan(neighbor_val):
                    neighbor_nan_count += 1
                else:
                    neighbor_count += 1
        
        if is_boundary:
            boundary_nans += 1
        elif neighbor_count == 0:  # All neighbors are NaN
            clustered_nans += 1
        else:
            isolated_nans += 1
    
    print(f"    Analysis of {len(nan_indices)} unfilled NaNs:")
    print(f"      At boundaries: {boundary_nans}")
    print(f"      In NaN clusters (all neighbors are NaN): {clustered_nans}")
    print(f"      Isolated (should have been fillable): {isolated_nans}")
    
    if isolated_nans > 0:
        print(f"    Warning: {isolated_nans} NaNs appear to be isolated but weren't filled - this might indicate a bug.")


def load_and_apply_patch_grid(combined_grid, patch_filename=None):
    """
    Load a separate likelihood grid (patch) and replace the [:, 19, :, 19, :] values 
    of the original combined likelihood grid with values from the patch grid.
    
    Parameters:
        combined_grid (np.ndarray): The original combined likelihood grid with shape (50, 20, 20, 20, 20).
        patch_filename (str, optional): Path to the patch file. If None, will search for patch files automatically.
        
    Returns:
        np.ndarray: The combined grid with patched values.
    """
    print("\n" + "="*60)
    print("APPLYING PATCH GRID")
    print("="*60)
    
    # If no patch filename provided, search for patch files
    if patch_filename is None:
        # Look for files with "alpha_idx19_patch" in the name
        patch_pattern = "*alpha_idx19_patch*.npy"
        patch_files = glob.glob(patch_pattern)
        
        if not patch_files:
            print(f"No patch files found matching pattern: {patch_pattern}")
            print("Skipping patch application.")
            return combined_grid
        elif len(patch_files) > 1:
            print(f"Multiple patch files found: {patch_files}")
            print(f"Using the first one: {patch_files[0]}")
            patch_filename = patch_files[0]
        else:
            patch_filename = patch_files[0]
            print(f"Found patch file: {patch_filename}")
    
    # Load the patch grid
    try:
        print(f"Loading patch grid from: {patch_filename}")
        patch_grid = np.load(patch_filename)
        print(f"Patch grid shape: {patch_grid.shape}")
    except Exception as e:
        print(f"Error loading patch file {patch_filename}: {e}")
        print("Skipping patch application.")
        return combined_grid
    
    # Check the current values at the problematic indices
    problematic_slice = combined_grid[:, 19, :, 19, :]
    problematic_nans = np.sum(np.isnan(problematic_slice))
    problematic_total = problematic_slice.size
    
    print(f"Problematic slice [:, 19, :, 19, :] analysis:")
    print(f"  Total values: {problematic_total}")
    print(f"  NaN values: {problematic_nans}")
    print(f"  Valid values: {problematic_total - problematic_nans}")
    
    # Check patch values at the same indices
    patch_slice = patch_grid[:, 19, :, 0, :]
    patch_nans = np.sum(np.isnan(patch_slice))
    patch_valid = np.sum(~np.isnan(patch_slice))

    print(f"Patch slice [:, 19, :, 0, :] analysis:")
    print(f"  Total values: {patch_slice.size}")
    print(f"  NaN values: {patch_nans}")
    print(f"  Valid values: {patch_valid}")
    
    # Apply the patch
    patched_grid = combined_grid.copy()
    
    print(f"Replacing values at indices [:, 19, :, 19, :] with patch values...")
    
    # Replace the entire slice
    patched_grid[:, 19, :, 19, :] = patch_slice
    
    # Verify the replacement
    new_slice = patched_grid[:, 19, :, 19, :]
    new_nans = np.sum(np.isnan(new_slice))
    new_valid = np.sum(~np.isnan(new_slice))
    
    print(f"After patching - slice [:, 19, :, 19, :] analysis:")
    print(f"  Total values: {new_slice.size}")
    print(f"  NaN values: {new_nans}")
    print(f"  Valid values: {new_valid}")
    
    # Calculate improvement
    nans_filled = problematic_nans - new_nans
    print(f"Patch results:")
    print(f"  NaNs filled: {nans_filled}")
    print(f"  Net improvement: {nans_filled} fewer NaNs")
    
    # Check if any values were actually different between original and patch
    if problematic_nans == 0:
        # Compare actual values if no NaNs in original
        original_values = combined_grid[:, 19, :, 19, :]
        patch_values = patch_grid[:, 19, :, 19, :]
        mask = ~(np.isnan(original_values) | np.isnan(patch_values))
        differences = np.sum(np.abs(original_values[mask] - patch_values[mask]) > 1e-10)
        print(f"  Values changed: {differences}")
    
    print("Patch application completed.")
    print("="*60)
    
    return patched_grid


def combine_likelihood_chunks(patch_file=None):
    """
    Combine the five likelihood grid chunks into a single file.
    
    Parameters:
        patch_file (str, optional): Path to a specific patch file to use. If None, will auto-detect.
    """
    
    # Define the base pattern for the chunk files
    base_pattern = "likelihood_grid_50mocktruths_90am_49stellar_alphaproxy_-1.571_1.571_scatter_0.010_1.000_x_0.010_0.950"
    
    # Define the chunk ranges in order
    chunk_ranges = ["0to4", "5to9", "10to14", "15to17", "18to19"]
    
    # List to store the loaded arrays
    chunks = []
    
    print("Loading likelihood grid chunks...")
    
    for chunk_range in chunk_ranges:
        filename = f"{base_pattern}_xchunk_{chunk_range}_nu_-3.000_3.000.npy"
        filepath = os.path.join(".", filename)
        
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            chunk_data = np.load(filepath)
            chunks.append(chunk_data)
            print(f"  Shape: {chunk_data.shape}")
        else:
            print(f"Warning: File {filename} not found!")
            return None
    
    if len(chunks) != 5:
        print(f"Error: Expected 5 chunks, but found {len(chunks)}")
        return None
    
    print("\nCombining chunks...")
    
    # Combine the chunks along the appropriate axis (axis=3 for x parameter)
    combined_grid = np.concatenate(chunks, axis=3)
    
    print(f"Combined shape: {combined_grid.shape}")
    
    # Check for NaNs before processing
    total_nans = np.sum(np.isnan(combined_grid))
    print(f"Total NaN values in combined grid: {total_nans}")
    
    # Apply patch grid to replace problematic values at [:, 19, :, 19, :] before filling
    combined_grid_patched = load_and_apply_patch_grid(combined_grid, patch_file)
    
    # Check NaNs after patching
    total_nans_after_patch = np.sum(np.isnan(combined_grid_patched))
    print(f"Total NaN values after patch application: {total_nans_after_patch}")
    
    if total_nans_after_patch > 0:
        print("\nFilling remaining NaN values for each mock dataset individually...")
        combined_grid_filled = fill_nans_iteratively_for_all_mocks(combined_grid_patched)
        
        # Check how many NaNs remain
        remaining_nans = np.sum(np.isnan(combined_grid_filled))
        print(f"Remaining NaN values after filling: {remaining_nans}")
    else:
        print("No NaN values found after patching, using patched grid.")
        combined_grid_filled = combined_grid_patched
    
    # Create output filenames
    output_filename_filled = f"{base_pattern}_nu_-3.000_3.000.npy"
    
    print(f"Saving patched + NaN-filled combined grid to {output_filename_filled}...")
    np.save(output_filename_filled, combined_grid_filled)
    
    print("Done!")
    
    return output_filename_filled

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Combine likelihood grid chunks, apply patches, and fill NaN values."
    )
    parser.add_argument(
        "--patch-file", 
        type=str, 
        default="/Users/fedorboreiko/Documents/Oxford/btfr_z/btfr/likelihood_grid_50mocktruths_90am_249stellar_alphaproxy_-1.571_1.571_scatter_0.010_1.000_x_0.010_0.950_xchunk_19to19_alpha_idx19_patch_nu_-3.000_3.000.npy",
        help="Path to a specific patch file to use for replacing [:, 19, :, 19, :] values. If not specified, will use the default patch file."
    )
    
    args = parser.parse_args()
    
    # Change to the btfr directory if not already there
    if not os.path.basename(os.getcwd()) == "btfr":
        if os.path.exists("btfr"):
            os.chdir("btfr")
            print("Changed directory to btfr/")
        else:
            print("Error: btfr directory not found!")
            exit(1)
    
    combine_likelihood_chunks(patch_file=args.patch_file)
