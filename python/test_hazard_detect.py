"""Tests for selected functions in hazard_detect.py."""
from hazard_detect import is_hazardous, linearRBG_to_Ls, sRBG_to_linearRGB, is_saturated_red_flash, is_luminance_flash
import numpy as np
from pytest import approx


def test_relative_luminance():
    """Test the relative linearRBG_to_Ls, sRBG_to_linearRGB functions."""
    color1 = np.array([255/255, 0, 0])
    color2 = np.array([245/255, 0, 0])
    Ls1 = linearRBG_to_Ls(sRBG_to_linearRGB(color1))
    Ls2 = linearRBG_to_Ls(sRBG_to_linearRGB(color2))

    assert approx(Ls1, rel=1e-3) == 0.2126
    assert approx(Ls2, rel=1e-3) == 0.1941


def test_is_hazardous_luminance():
    """Test the for a hazardous flash that satisfies the luminance condition. """
    color1 = np.array([0, 0, 0])
    color2 = np.array([1, 1, 1])
    assert is_hazardous(color1, color2) == True
    assert is_hazardous(color2, color1) == True

    color1 = np.array([0.5, 0.5, 0.5])
    color2 = np.array([1, 1, 1])
    assert is_hazardous(color1, color2) == True
    assert is_hazardous(color2, color1) == True


def test_is_hazardous_saturated_red_and_not_luminous():
    """Test the for a hazrdous flash that satisfies the saturated red condition but not the luminance condition. """
    color1 = np.array([255/255, 0, 0])
    color2 = np.array([245/255, 0, 0])
    assert is_hazardous(color1, color2) == True
    assert is_hazardous(color2, color1) == True


def test_is_luminous_flash():
    """Test the for a luminous flash."""

    # Negative test cases
    color1 = np.array([255/255, 0, 0])
    color2 = np.array([245/255, 0, 0])
    Ls1 = linearRBG_to_Ls(sRBG_to_linearRGB(color1))
    Ls2 = linearRBG_to_Ls(sRBG_to_linearRGB(color2))
    assert is_luminance_flash(Ls1, Ls2) == False
    assert is_luminance_flash(Ls2, Ls1) == False

    # Postive test cases
    color1 = np.array([1, 0, 0])
    color2 = np.array([0, 0, 0])
    Ls1 = linearRBG_to_Ls(sRBG_to_linearRGB(color1))
    Ls2 = linearRBG_to_Ls(sRBG_to_linearRGB(color2))
    assert is_luminance_flash(Ls1, Ls2) == True
    assert is_luminance_flash(Ls2, Ls1) == True

    color1 = np.array([1, 0, 0])
    color2 = np.array([0.7, 0.7, 0.7])
    Ls1 = linearRBG_to_Ls(sRBG_to_linearRGB(color1))
    Ls2 = linearRBG_to_Ls(sRBG_to_linearRGB(color2))
    assert is_luminance_flash(Ls1, Ls2) == True
    assert is_luminance_flash(Ls2, Ls1) == True


def test_is_saturated_red_flash():
    """Test the for a saturated red flash."""

    # Positive test cases
    color1 = np.array([255/255, 0, 0])
    color2 = np.array([245/255, 0, 0])
    linear_color1 = sRBG_to_linearRGB(color1)
    linear_color2 = sRBG_to_linearRGB(color2)
    assert is_saturated_red_flash(linear_color1, linear_color2) == True
    assert is_saturated_red_flash(linear_color2, linear_color1) == True

    # Negative test cases
    color1 = np.array([0, 255/255, 0])
    color2 = np.array([0, 245/255, 0])
    linear_color1 = sRBG_to_linearRGB(color1)
    linear_color2 = sRBG_to_linearRGB(color2)
    assert is_saturated_red_flash(linear_color1, linear_color2) == False
    assert is_saturated_red_flash(linear_color2, linear_color1) == False
