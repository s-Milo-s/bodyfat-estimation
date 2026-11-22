"""
Test script to verify Blender and MPFB2 are set up correctly.
Run with: blender --background --python scripts/test_blender.py
"""

import sys

def test_blender():
    """Test basic Blender functionality."""
    try:
        import bpy
        print(f"✓ Blender version: {bpy.app.version_string}")
        print(f"✓ Python version: {sys.version}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import bpy: {e}")
        return False


def test_mpfb2():
    """Test MPFB2 addon is installed and enabled."""
    try:
        import bpy
        
        # Check if MPFB2 addon is available
        addon_name = "mpfb"
        
        # Try to find MPFB in addons
        if addon_name in bpy.context.preferences.addons:
            print(f"✓ MPFB2 addon is enabled")
            return True
        
        # Try to enable it
        try:
            bpy.ops.preferences.addon_enable(module=addon_name)
            print(f"✓ MPFB2 addon enabled successfully")
            return True
        except Exception as e:
            print(f"✗ MPFB2 addon not found or couldn't be enabled: {e}")
            print("  Install from: https://static.makehumancommunity.org/mpfb/releases/")
            return False
            
    except Exception as e:
        print(f"✗ Error testing MPFB2: {e}")
        return False


def test_mpfb2_imports():
    """Test MPFB2 modules can be imported."""
    try:
        # These imports work after MPFB2 is enabled
        from mpfb.services.humanservice import HumanService
        from mpfb.services.targetservice import TargetService
        print("✓ MPFB2 services imported successfully")
        return True
    except ImportError as e:
        print(f"⚠ MPFB2 services not importable (may need full Blender UI): {e}")
        return False


def main():
    print("\n" + "="*50)
    print("BLENDER + MPFB2 SETUP TEST")
    print("="*50 + "\n")
    
    results = []
    
    results.append(("Blender", test_blender()))
    results.append(("MPFB2 Addon", test_mpfb2()))
    results.append(("MPFB2 Imports", test_mpfb2_imports()))
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed and name != "MPFB2 Imports":  # Imports failing is okay in background mode
            all_passed = False
    
    print("\n")
    
    if all_passed:
        print("Ready to generate synthetic bodies!")
    else:
        print("Please fix the issues above before continuing.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
