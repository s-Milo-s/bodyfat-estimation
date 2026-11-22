"""
Extract body measurements from MPFB2 mesh by slicing at anatomical landmarks.

Measurements extracted:
- Neck circumference
- Chest circumference  
- Waist circumference
- Hip circumference

This runs inside Blender with an active mesh object.
"""

import bpy
import bmesh
import math
from mathutils import Vector
from mathutils.geometry import intersect_line_plane


def get_mesh_height_range(obj):
    """Get the min/max Z coordinates of the mesh."""
    mesh = obj.data
    world_matrix = obj.matrix_world
    
    z_coords = []
    for vert in mesh.vertices:
        world_pos = world_matrix @ vert.co
        z_coords.append(world_pos.z)
    
    return min(z_coords), max(z_coords)


def slice_mesh_at_height(obj, z_height):
    """
    Slice mesh at a given height and return the circumference.
    
    Uses raycasting approach: cast rays from center outward at the slice plane,
    find intersection points, compute perimeter.
    """
    mesh = obj.data
    world_matrix = obj.matrix_world
    
    # Get all edges that cross this Z height
    crossing_points = []
    
    for edge in mesh.edges:
        v1_idx, v2_idx = edge.vertices
        v1_world = world_matrix @ mesh.vertices[v1_idx].co
        v2_world = world_matrix @ mesh.vertices[v2_idx].co
        
        z1, z2 = v1_world.z, v2_world.z
        
        # Check if edge crosses the slice plane
        if (z1 <= z_height <= z2) or (z2 <= z_height <= z1):
            if abs(z2 - z1) > 0.0001:  # Avoid division by zero
                # Interpolate to find crossing point
                t = (z_height - z1) / (z2 - z1)
                point = v1_world.lerp(v2_world, t)
                crossing_points.append((point.x, point.y))
    
    if len(crossing_points) < 3:
        return None
    
    # Sort points by angle from center to form a closed polygon
    cx = sum(p[0] for p in crossing_points) / len(crossing_points)
    cy = sum(p[1] for p in crossing_points) / len(crossing_points)
    
    def angle_from_center(p):
        return math.atan2(p[1] - cy, p[0] - cx)
    
    sorted_points = sorted(crossing_points, key=angle_from_center)
    
    # Calculate perimeter (circumference)
    circumference = 0
    for i in range(len(sorted_points)):
        p1 = sorted_points[i]
        p2 = sorted_points[(i + 1) % len(sorted_points)]
        dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        circumference += dist
    
    return circumference


def find_narrowest_slice(obj, z_min, z_max, num_samples=50):
    """Find the Z height with the smallest circumference in a range."""
    best_z = None
    best_circ = float('inf')
    
    for i in range(num_samples):
        z = z_min + (z_max - z_min) * i / (num_samples - 1)
        circ = slice_mesh_at_height(obj, z)
        if circ is not None and circ < best_circ:
            best_circ = circ
            best_z = z
    
    return best_z, best_circ


def find_widest_slice(obj, z_min, z_max, num_samples=50):
    """Find the Z height with the largest circumference in a range."""
    best_z = None
    best_circ = 0
    
    for i in range(num_samples):
        z = z_min + (z_max - z_min) * i / (num_samples - 1)
        circ = slice_mesh_at_height(obj, z)
        if circ is not None and circ > best_circ:
            best_circ = circ
            best_z = z
    
    return best_z, best_circ


def extract_body_measurements(obj):
    """
    Extract all body measurements from a human mesh.
    
    Assumes mesh is standing upright with Z as vertical axis.
    Returns measurements in centimeters.
    """
    z_min, z_max = get_mesh_height_range(obj)
    height = z_max - z_min
    
    # Define anatomical regions as fractions of body height
    # These are approximate and may need tuning for MPFB2 meshes
    
    # Neck: around 85-90% of height
    neck_z_min = z_min + height * 0.83
    neck_z_max = z_min + height * 0.88
    
    # Chest: around 70-80% of height (at armpit/bust level)
    chest_z_min = z_min + height * 0.68
    chest_z_max = z_min + height * 0.78
    
    # Waist: around 55-65% of height (narrowest part of torso)
    waist_z_min = z_min + height * 0.55
    waist_z_max = z_min + height * 0.65
    
    # Hip: around 45-55% of height (widest part of pelvis)
    hip_z_min = z_min + height * 0.45
    hip_z_max = z_min + height * 0.55
    
    # Extract measurements
    measurements = {}
    
    # Neck - find narrowest point in neck region
    neck_z, neck_circ = find_narrowest_slice(obj, neck_z_min, neck_z_max, num_samples=20)
    measurements['neck_cm'] = round(neck_circ * 100, 1) if neck_circ else None
    measurements['neck_z'] = neck_z
    
    # Chest - find widest point in chest region
    chest_z, chest_circ = find_widest_slice(obj, chest_z_min, chest_z_max, num_samples=30)
    measurements['chest_cm'] = round(chest_circ * 100, 1) if chest_circ else None
    measurements['chest_z'] = chest_z
    
    # Waist - find narrowest point in waist region
    waist_z, waist_circ = find_narrowest_slice(obj, waist_z_min, waist_z_max, num_samples=30)
    measurements['waist_cm'] = round(waist_circ * 100, 1) if waist_circ else None
    measurements['waist_z'] = waist_z
    
    # Hip - find widest point in hip region
    hip_z, hip_circ = find_widest_slice(obj, hip_z_min, hip_z_max, num_samples=30)
    measurements['hip_cm'] = round(hip_circ * 100, 1) if hip_circ else None
    measurements['hip_z'] = hip_z
    
    # Height
    measurements['height_cm'] = round(height * 100, 1)
    
    return measurements


def calculate_body_fat_navy(measurements, sex):
    """
    Calculate body fat percentage using the US Navy formula.
    
    Men: %BF = 86.010 × log10(waist - neck) - 70.041 × log10(height) + 36.76
    Women: %BF = 163.205 × log10(waist + hip - neck) - 97.684 × log10(height) - 78.387
    
    Args:
        measurements: dict with waist_cm, hip_cm, neck_cm, height_cm
        sex: "male" or "female"
    
    Returns:
        Body fat percentage or None if measurements incomplete
    """
    waist = measurements.get('waist_cm')
    neck = measurements.get('neck_cm')
    height = measurements.get('height_cm')
    hip = measurements.get('hip_cm')
    
    if not all([waist, neck, height]):
        return None
    
    if sex == "male":
        if waist <= neck:
            return None
        bf = 86.010 * math.log10(waist - neck) - 70.041 * math.log10(height) + 36.76
    else:  # female
        if not hip or (waist + hip) <= neck:
            return None
        bf = 163.205 * math.log10(waist + hip - neck) - 97.684 * math.log10(height) - 78.387
    
    # Clamp to reasonable range
    bf = max(5, min(60, bf))
    return round(bf, 1)


# Test function when run directly in Blender
if __name__ == "__main__":
    # Get active object
    obj = bpy.context.active_object
    
    if obj and obj.type == 'MESH':
        print("\nExtracting measurements from:", obj.name)
        measurements = extract_body_measurements(obj)
        
        print("\nMeasurements:")
        for key, value in measurements.items():
            if not key.endswith('_z'):
                print(f"  {key}: {value}")
        
        # Try to calculate body fat (assume female for test)
        bf = calculate_body_fat_navy(measurements, "female")
        if bf:
            print(f"\nEstimated body fat (Navy formula): {bf}%")
    else:
        print("No mesh object selected")
