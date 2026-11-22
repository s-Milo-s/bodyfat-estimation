"""
Generate a synthetic human body and render front + side views.
Run with: blender --background --python scripts/generate_body.py -- --output data/renders/test

This script uses MPFB2 to generate a human body with specified parameters,
poses it in A-pose, and renders front and side views.
"""

import bpy
import bmesh
import sys
import os
import argparse
import json
import math
import random
from mathutils import Vector


# === ENABLE MPFB2 EXTENSION ===

def enable_mpfb2():
    """Enable MPFB2 extension if available."""
    try:
        # Try to enable the extension
        bpy.ops.preferences.addon_enable(module="bl_ext.blender_org.mpfb")
        print("✓ MPFB2 extension enabled")
        return True
    except Exception as e1:
        try:
            # Try alternate name
            bpy.ops.preferences.addon_enable(module="mpfb")
            print("✓ MPFB2 addon enabled")
            return True
        except Exception as e2:
            print(f"Could not enable MPFB2: {e1}, {e2}")
            return False

# Try to enable MPFB2 at startup
enable_mpfb2()


# === CONFIGURATION ===

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 682  # 3:4 portrait ratio (iPhone selfie)

# Camera settings - framed on torso (head to mid-thigh)
CAMERA_DISTANCE = 2.5  # meters from subject (closer for tighter frame)
CAMERA_HEIGHT = 1.1    # meters (chest height for torso focus)

# Lighting
LIGHT_INTENSITY = 1000  # watts


# === BODY PARAMETER SAMPLING ===

def sample_body_params(seed=None, override_bmi=None, override_sex=None):
    """
    Sample realistic body parameters from approximate population distributions.
    Returns dict of parameters for body generation.
    
    Args:
        seed: Random seed for reproducibility
        override_bmi: Force specific BMI value (18-45)
        override_sex: Force "male" or "female"
    """
    if seed is not None:
        random.seed(seed)
    
    # Sample or override sex
    sex = override_sex if override_sex else random.choice(["male", "female"])
    
    # Sample age (18-70, weighted toward middle)
    age = int(random.gauss(40, 15))
    age = max(18, min(70, age))
    
    # Sample height based on sex (in cm)
    if sex == "male":
        height = random.gauss(175, 7)  # US male average ~175cm
    else:
        height = random.gauss(162, 6)  # US female average ~162cm
    height = max(150, min(200, height))
    
    # Sample or override BMI
    if override_bmi is not None:
        bmi = max(16, min(50, override_bmi))  # Allow extreme values when overriding
    else:
        bmi = random.gauss(26, 6)  # US average BMI ~26
        bmi = max(18.5, min(45, bmi))
    
    # Calculate weight from height and BMI
    height_m = height / 100
    weight = bmi * (height_m ** 2)
    
    return {
        "sex": sex,
        "age": age,
        "height_cm": round(height, 1),
        "weight_kg": round(weight, 1),
        "bmi": round(bmi, 1),
    }


# === SCENE SETUP ===

def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def setup_lighting():
    """Set up basic 3-point lighting."""
    # Key light (main)
    bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
    key_light = bpy.context.active_object
    key_light.name = "KeyLight"
    key_light.data.energy = LIGHT_INTENSITY
    key_light.data.size = 2
    
    # Fill light (softer, opposite side)
    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 2))
    fill_light = bpy.context.active_object
    fill_light.name = "FillLight"
    fill_light.data.energy = LIGHT_INTENSITY * 0.5
    fill_light.data.size = 3
    
    # Back light (rim)
    bpy.ops.object.light_add(type='AREA', location=(0, 2, 3))
    back_light = bpy.context.active_object
    back_light.name = "BackLight"
    back_light.data.energy = LIGHT_INTENSITY * 0.3
    back_light.data.size = 2


def setup_camera(view="front"):
    """
    Set up camera for front or side view.
    
    Args:
        view: "front" or "side"
    """
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = f"Camera_{view}"
    
    if view == "front":
        camera.location = (0, -CAMERA_DISTANCE, CAMERA_HEIGHT)
        camera.rotation_euler = (math.radians(90), 0, 0)
    elif view == "side":
        camera.location = (CAMERA_DISTANCE, 0, CAMERA_HEIGHT)
        camera.rotation_euler = (math.radians(90), 0, math.radians(90))
    
    bpy.context.scene.camera = camera
    return camera


def setup_render_settings():
    """Configure render settings."""
    scene = bpy.context.scene
    
    # Resolution
    scene.render.resolution_x = IMAGE_WIDTH
    scene.render.resolution_y = IMAGE_HEIGHT
    scene.render.resolution_percentage = 100
    
    # Use Cycles for better quality (or EEVEE for speed)
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64  # Lower for faster test renders
    scene.cycles.use_denoising = True
    
    # Transparent background (for easy segmentation)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    
    # Use GPU if available
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'METAL'  # macOS
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True


# === BODY GENERATION ===

def create_simple_body(params):
    """
    Create a simple human body mesh for testing.
    
    In production, this would use MPFB2's HumanService.
    For now, creates a basic humanoid shape to test the pipeline.
    """
    # For initial testing, create a simple capsule/cylinder humanoid
    # This will be replaced with MPFB2 body generation
    
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.15,
        depth=0.6,
        location=(0, 0, 0.9)  # Torso
    )
    torso = bpy.context.active_object
    torso.name = "Torso"
    
    # Head
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.12,
        location=(0, 0, 1.35)
    )
    head = bpy.context.active_object
    head.name = "Head"
    
    # Legs (simplified)
    for x_offset in [-0.08, 0.08]:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.08,
            depth=0.8,
            location=(x_offset, 0, 0.25)
        )
        leg = bpy.context.active_object
        leg.name = f"Leg_{'L' if x_offset < 0 else 'R'}"
    
    # Arms - selfie pose (right arm holds phone up, left arm relaxed at side)
    # Left arm - relaxed, slightly away from body
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.05,
        depth=0.5,
        location=(-0.35, 0, 0.85)
    )
    arm_l = bpy.context.active_object
    arm_l.name = "Arm_L"
    arm_l.rotation_euler = (0, 0, math.radians(-15))  # Slight angle out
    
    # Right arm - holding phone (bent up, out of frame)
    # Upper arm
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.05,
        depth=0.3,
        location=(0.35, -0.1, 1.15)
    )
    arm_r_upper = bpy.context.active_object
    arm_r_upper.name = "Arm_R_Upper"
    arm_r_upper.rotation_euler = (math.radians(-45), 0, math.radians(30))
    
    # Join all body parts (not lights)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.context.view_layer.objects.active = torso
    bpy.ops.object.join()
    
    body = bpy.context.active_object
    body.name = "Body"
    
    # Scale based on height parameter
    target_height = params["height_cm"] / 100  # Convert to meters
    current_height = 1.5  # Approximate height of our simple model
    scale_factor = target_height / current_height
    body.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    bpy.ops.object.transform_apply(scale=True)
    
    # Add a simple material
    mat = bpy.data.materials.new(name="SkinMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.8, 0.6, 0.5, 1)  # Skin tone
    body.data.materials.append(mat)
    
    return body


def create_mpfb2_body(params):
    """
    Create a human body using MPFB2.
    
    This is the production method - requires MPFB2 addon to be installed.
    """
    try:
        # Try extension path first (Blender 4.2+)
        try:
            from bl_ext.blender_org.mpfb.services.humanservice import HumanService
            from bl_ext.blender_org.mpfb.services.targetservice import TargetService
        except ImportError:
            # Fall back to addon path
            from mpfb.services.humanservice import HumanService
            from mpfb.services.targetservice import TargetService
        
        # Build macro_detail_dict for body shape variation
        # gender: 0.0 = female, 1.0 = male
        # age: 0.0 = baby, 0.25 = child, 0.5 = young, 0.75 = old, 1.0 = very old
        # weight: 0.0 = thin, 0.5 = average, 1.0 = heavy
        # muscle: 0.0 = weak, 0.5 = average, 1.0 = muscular
        
        gender_value = 1.0 if params["sex"] == "male" else 0.0
        
        # Map age 18-70 to MPFB's young-old range (0.4 to 0.8)
        age_normalized = 0.4 + max(0, min(1, (params["age"] - 18) / (70 - 18))) * 0.4
        
        # Map BMI to weight: BMI 18.5 -> 0.1, BMI 25 -> 0.5, BMI 40 -> 1.0
        bmi = params["bmi"]
        weight_normalized = max(0.0, min(1.0, (bmi - 18.5) / (40 - 18.5)))
        
        # Muscle inversely related to fat (simplification)
        muscle_normalized = max(0.2, min(0.8, 0.6 - weight_normalized * 0.3))
        
        # Random race distribution
        import random
        asian = random.random()
        african = random.random() 
        caucasian = random.random()
        total = asian + african + caucasian
        
        macro_detail_dict = {
            "gender": gender_value,
            "age": age_normalized,
            "muscle": muscle_normalized,
            "weight": weight_normalized,
            "proportions": 0.5,
            "height": 0.5,  # We'll scale afterwards
            "cupsize": 0.5,
            "firmness": 0.5,
            "race": {
                "asian": asian / total,
                "caucasian": caucasian / total,
                "african": african / total
            }
        }
        
        print(f"  Macro targets: gender={gender_value}, age={age_normalized:.2f}, weight={weight_normalized:.2f}, muscle={muscle_normalized:.2f}")
        
        # Create human with macro details
        basemesh = HumanService.create_human(macro_detail_dict=macro_detail_dict)
        
        if basemesh is None:
            print("Failed to create MPFB2 human, falling back to simple body")
            return create_simple_body(params)
        
        body = basemesh
        body.name = "Body"
        
        # Apply detailed body targets for realistic fat distribution
        # Higher BMI = more belly, wider waist/hips, less muscle tone
        if weight_normalized > 0.2:  # Apply fat details for overweight bodies
            fat_amount = min(1.0, (weight_normalized - 0.2) / 0.6)  # 0-1 scale, hits max earlier
            
            detail_targets = [
                # BELLY - MAXED OUT
                ("stomach/stomach-pregnant-incr", fat_amount * 3.0),  # Huge belly bulge
                ("stomach/stomach-tone-decr", fat_amount * 2.0),      # Very flabby
                ("stomach/stomach-navel-out", fat_amount * 1.5),      # Navel pushed way out
                
                # WAIST/HIPS - love handles area
                ("torso/measure-waist-circ-incr", fat_amount * 2.5),  # Much wider waist
                ("torso/measure-hips-circ-incr", fat_amount * 2.0),   # Much wider hips
                ("hip/hip-scale-depth-incr", fat_amount * 1.5),       # Deeper hips front-back
                ("hip/hip-scale-horiz-incr", fat_amount * 1.5),       # Wider hips side-side
                
                # BUTT
                ("buttocks/buttocks-volume-incr", fat_amount * 1.5),  # Much bigger butt
                
                # CHEST (for men = chest fat, for women = larger bust handled by cupsize)
                ("torso/measure-bust-circ-incr", fat_amount * 1.0 if gender_value > 0.5 else 0),
            ]
            
            print(f"  Applying fat detail targets (fat_amount={fat_amount:.2f})")
            for target_name, value in detail_targets:
                if value > 0:
                    try:
                        TargetService.set_target_value(body, target_name, value)
                        print(f"    {target_name} = {value:.2f}")
                    except Exception as e:
                        print(f"    Warning: Could not apply {target_name}: {e}")
        
        # Apply height by scaling
        # MPFB2 basemesh is roughly 1.68m tall
        target_height = params["height_cm"] / 100
        base_height = 1.68
        scale_factor = target_height / base_height
        body.scale = (scale_factor, scale_factor, scale_factor)
        
        bpy.context.view_layer.objects.active = body
        body.select_set(True)
        bpy.ops.object.transform_apply(scale=True)
        
        # Add a simple skin material
        if len(body.data.materials) == 0:
            mat = bpy.data.materials.new(name="SkinMaterial")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]
            # Skin tone based on random variation
            skin_base = (0.8, 0.6, 0.5, 1)
            bsdf.inputs["Base Color"].default_value = skin_base
            bsdf.inputs["Roughness"].default_value = 0.5
            body.data.materials.append(mat)
        
        return body
        
    except ImportError as e:
        print(f"MPFB2 import error: {e}")
        print("Falling back to simple body placeholder")
        return create_simple_body(params)
    except Exception as e:
        print(f"MPFB2 error: {e}")
        print("Falling back to simple body placeholder")
        return create_simple_body(params)


# === MEASUREMENT EXTRACTION ===

def get_mesh_height_range(obj):
    """Get the min/max Z coordinates of the mesh."""
    mesh = obj.data
    world_matrix = obj.matrix_world
    
    z_coords = []
    for vert in mesh.vertices:
        world_pos = world_matrix @ vert.co
        z_coords.append(world_pos.z)
    
    return min(z_coords), max(z_coords)


def slice_mesh_at_height(obj, z_height, torso_width_limit=0.25):
    """
    Slice mesh at a given height and return the circumference.
    Uses ellipse approximation from width/depth of torso region only.
    
    Args:
        obj: Blender mesh object
        z_height: Height to slice at (meters)
        torso_width_limit: Max X distance from center to include (meters)
                          Filters out arms which extend beyond this
    """
    mesh = obj.data
    world_matrix = obj.matrix_world
    
    # Find all vertices near this Z height
    tolerance = 0.02  # 2cm tolerance
    nearby_verts = []
    
    for vert in mesh.vertices:
        world_pos = world_matrix @ vert.co
        if abs(world_pos.z - z_height) < tolerance:
            nearby_verts.append((world_pos.x, world_pos.y, world_pos.z))
    
    if len(nearby_verts) < 4:
        return None
    
    # Filter to torso only (exclude arms by X distance from center)
    # Torso is centered at X ≈ 0
    torso_verts = [v for v in nearby_verts if abs(v[0]) < torso_width_limit]
    
    if len(torso_verts) < 4:
        # Fallback: use all verts but be more selective
        torso_verts = nearby_verts
    
    # Get width (X extent) and depth (Y extent) of torso at this height
    x_coords = [v[0] for v in torso_verts]
    y_coords = [v[1] for v in torso_verts]
    
    width = max(x_coords) - min(x_coords)   # Side to side
    depth = max(y_coords) - min(y_coords)   # Front to back
    
    # Approximate circumference using ellipse formula
    # Ramanujan approximation: π * (3(a+b) - sqrt((3a+b)(a+3b)))
    a = width / 2  # semi-major axis
    b = depth / 2  # semi-minor axis
    
    circumference = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    
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
    
    return best_z, best_circ if best_circ != float('inf') else None


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
    
    return best_z, best_circ if best_circ > 0 else None


def extract_measurements(obj):
    """
    Extract body measurements from a human mesh by slicing at anatomical landmarks.
    Returns measurements in centimeters.
    """
    z_min, z_max = get_mesh_height_range(obj)
    height = z_max - z_min
    
    # Define anatomical regions as fractions of body height
    # Neck: around 88-92% of height (higher up, below the jaw)
    neck_z_min = z_min + height * 0.88
    neck_z_max = z_min + height * 0.92
    
    # Chest: around 70-80% of height (at armpit/bust level)
    chest_z_min = z_min + height * 0.68
    chest_z_max = z_min + height * 0.78
    
    # Waist: around 55-65% of height (narrowest part of torso)
    waist_z_min = z_min + height * 0.55
    waist_z_max = z_min + height * 0.65
    
    # Hip: around 45-55% of height (widest part of pelvis)
    hip_z_min = z_min + height * 0.45
    hip_z_max = z_min + height * 0.55
    
    measurements = {}
    
    # Neck - find narrowest point in neck region (use tighter torso filter for neck)
    def slice_neck(z):
        return slice_mesh_at_height(obj, z, torso_width_limit=0.10)  # Narrower for neck
    
    best_neck_circ = float('inf')
    for i in range(20):
        z = neck_z_min + (neck_z_max - neck_z_min) * i / 19
        circ = slice_neck(z)
        if circ and circ < best_neck_circ:
            best_neck_circ = circ
    
    neck_circ = best_neck_circ if best_neck_circ != float('inf') else None
    measurements['neck_cm'] = round(neck_circ * 100, 1) if neck_circ else None
    
    # Chest - find widest point in chest region
    _, chest_circ = find_widest_slice(obj, chest_z_min, chest_z_max, num_samples=30)
    measurements['chest_cm'] = round(chest_circ * 100, 1) if chest_circ else None
    
    # Waist - find narrowest point in waist region
    _, waist_circ = find_narrowest_slice(obj, waist_z_min, waist_z_max, num_samples=30)
    measurements['waist_cm'] = round(waist_circ * 100, 1) if waist_circ else None
    
    # Hip - find widest point in hip region
    _, hip_circ = find_widest_slice(obj, hip_z_min, hip_z_max, num_samples=30)
    measurements['hip_cm'] = round(hip_circ * 100, 1) if hip_circ else None
    
    # Height
    measurements['height_cm'] = round(height * 100, 1)
    
    return measurements


# === RENDERING ===

def render_view(output_path, view="front"):
    """Render a single view and save to file."""
    # Remove existing cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj)
    
    # Set up camera for this view
    setup_camera(view)
    
    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    print(f"Rendered {view} view to: {output_path}")


# === MAIN ===

def parse_args():
    """Parse command line arguments after '--'."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="Generate synthetic body renders")
    parser.add_argument("--output", "-o", required=True, help="Output path prefix")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    parser.add_argument("--use-mpfb2", action="store_true", help="Use MPFB2 (requires addon)")
    parser.add_argument("--bmi", type=float, default=None, help="Override BMI (18-45)")
    parser.add_argument("--sex", choices=["male", "female"], default=None, help="Override sex")
    
    return parser.parse_args(argv)


def main():
    args = parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "="*50)
    print("SYNTHETIC BODY GENERATION")
    print("="*50 + "\n")
    
    # Sample body parameters
    params = sample_body_params(seed=args.seed, override_bmi=args.bmi, override_sex=args.sex)
    print(f"Body parameters: {json.dumps(params, indent=2)}")
    
    # Set up scene
    print("\nSetting up scene...")
    clear_scene()
    setup_lighting()
    setup_render_settings()
    
    # Create body - try MPFB2 first, fall back to simple
    print("Creating body...")
    if args.use_mpfb2 or True:  # Always try MPFB2 first now
        body = create_mpfb2_body(params)
    else:
        body = create_simple_body(params)
    
    # Center body at origin
    body.location = (0, 0, 0)
    
    # Extract measurements
    print("Extracting measurements...")
    measurements = extract_measurements(body)
    print(f"Measurements: {json.dumps(measurements, indent=2)}")
    
    # Render views
    print("\nRendering views...")
    render_view(f"{args.output}_front.png", view="front")
    render_view(f"{args.output}_side.png", view="side")
    
    # Save metadata
    metadata = {
        "params": params,
        "measurements": measurements,
        "render_settings": {
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "engine": "CYCLES",
        }
    }
    
    metadata_path = f"{args.output}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    
    print("\n✓ Generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
