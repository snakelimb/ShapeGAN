import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.colors import hsv_to_rgb
import colorsys

def wrap_hue(h):
    """Wrap hue value to [0, 1] range"""
    return h % 1.0

def generate_organic_curve(start, end, control_intensity, rng, num_points=25):
    """Generate a curved line between two points using Bezier-like curves"""
    t = np.linspace(0, 1, num_points)

    # Calculate perpendicular direction for control points
    direction = end - start
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)

    # Create control points with organic variation
    mid = (start + end) / 2
    control1 = start + direction * 0.3 + perpendicular * control_intensity * rng.uniform(-1, 1)
    control2 = end - direction * 0.3 + perpendicular * control_intensity * rng.uniform(-1, 1)

    # Generate bezier curve
    curve_points = []
    for t_val in t:
        # Cubic bezier formula
        point = (1-t_val)**3 * start + 3*(1-t_val)**2*t_val * control1 + \
                3*(1-t_val)*t_val**2 * control2 + t_val**3 * end
        curve_points.append(point)

    return np.array(curve_points)

def create_organic_polygon(center, radius, num_points, irregularity, rng):
    """Create an irregular organic polygon"""
    angles = np.linspace(0, 2*np.pi, num_points + 1)[:-1]

    # Add angular irregularity
    angles += rng.uniform(-irregularity, irregularity, num_points)
    angles = np.sort(angles)

    points = []
    for angle in angles:
        # Vary radius organically
        r = radius * rng.uniform(0.7, 1.3)
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        points.append([x, y])

    return points

def apply_symmetry_organic(points, symmetry_type, symmetry_strength, center=[0, 0]):
    """Apply symmetry transformations while maintaining organic qualities"""
    if symmetry_strength < 0.1:
        return points

    if symmetry_type < 0.33:  # Radial symmetry
        # Force radial symmetry by averaging angular positions
        radial_points = []
        for point in points:
            angle = np.arctan2(point[1] - center[1], point[0] - center[0])
            radius = np.linalg.norm([point[0] - center[0], point[1] - center[1]])

            # Snap to nearest radial division
            num_divisions = 4 + int(symmetry_strength * 8)
            snapped_angle = round(angle * num_divisions / (2 * np.pi)) * (2 * np.pi) / num_divisions

            # Blend between original and snapped with organic variation
            blend_angle = angle * (1 - symmetry_strength) + snapped_angle * symmetry_strength

            new_x = center[0] + radius * np.cos(blend_angle)
            new_y = center[1] + radius * np.sin(blend_angle)
            radial_points.append([new_x, new_y])
        return radial_points

    elif symmetry_type < 0.66:  # Rococo (flowing asymmetric)
        # Add flowing, organic asymmetric variations
        rococo_points = []
        for i, point in enumerate(points):
            # Create flowing offset based on position
            flow_factor = np.sin(i * 0.5) * np.cos(i * 0.3)
            offset_magnitude = symmetry_strength * 0.8

            offset_x = flow_factor * offset_magnitude * np.sin(i * 2.1)
            offset_y = flow_factor * offset_magnitude * np.cos(i * 1.7)

            new_x = point[0] + offset_x
            new_y = point[1] + offset_y
            rococo_points.append([new_x, new_y])
        return rococo_points

    else:  # Bilateral symmetry
        # Force bilateral symmetry across vertical axis
        bilateral_points = []
        for point in points:
            # Mirror point across vertical axis through center
            mirrored_x = center[0] - (point[0] - center[0])

            # Blend between original and mirrored
            new_x = point[0] * (1 - symmetry_strength) + mirrored_x * symmetry_strength
            new_y = point[1]
            bilateral_points.append([new_x, new_y])
        return bilateral_points

class OrganicBiomorph:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_radial_biomorph(self, center, params, base_hsv, subtract_hsv):
        """Generate radially symmetric biomorph"""
        num_arms = 4 + int(params['arm_count'] * 8)  # 4-12 arms
        arm_length_base = 1.5 + params['arm_length'] * 2.0
        inner_radius = 0.3 + params['body_size'] * 0.5

        elements = []

        # Central body with organic shape
        body_points = int(6 + params['body_complexity'] * 10)
        central_body = create_organic_polygon(center, inner_radius, body_points,
                                            params['irregularity'] * 0.5, self.rng)
        body_color = hsv_to_rgb([base_hsv[0], base_hsv[1], base_hsv[2]])
        body_polygon = Polygon(central_body, color=body_color, alpha=0.8)
        elements.append(body_polygon)

        # Generate arms
        for i in range(num_arms):
            arm_activation = params['arm_activations'][i % len(params['arm_activations'])]
            if arm_activation < 0.6:
                continue

            angle = (2 * np.pi * i) / num_arms

            # Arm length with variation
            arm_length = arm_length_base * self.rng.uniform(0.7, 1.3)
            arm_length *= (1 + params['growth_variation'] * self.rng.uniform(-0.5, 0.5))

            # Calculate arm end position
            arm_dir = np.array([np.cos(angle), np.sin(angle)])
            arm_end = center + arm_dir * arm_length

            # Create arm segments
            num_segments = 3 + int(params['segmentation'] * 6)
            segment_length = arm_length / num_segments

            current_pos = center + arm_dir * inner_radius
            current_dir = arm_dir.copy()

            # Arm color (subtract color based on arm index)
            color_factor = i / num_arms
            arm_hsv = [
                wrap_hue(base_hsv[0] - subtract_hsv[0] * color_factor),
                max(0, min(1, base_hsv[1] - subtract_hsv[1] * color_factor)),
                max(0, min(1, base_hsv[2] - subtract_hsv[2] * color_factor))
            ]
            arm_color = hsv_to_rgb(arm_hsv)

            # Generate arm segments
            for seg in range(num_segments):
                # Segment width (tapers toward tip)
                width_factor = 1.0 - (seg / num_segments) * 0.7
                segment_width = (0.1 + params['arm_width'] * 0.2) * width_factor

                # Add organic variation to direction
                current_dir += self.rng.normal(0, params['curvature'] * 0.3, 2)
                current_dir = current_dir / np.linalg.norm(current_dir)

                # Calculate next position
                next_pos = current_pos + current_dir * segment_length

                # Create segment as organic curve
                control_intensity = params['curvature'] * 0.5
                if seg > 0:  # Don't curve the first segment too much
                    curve_points = generate_organic_curve(current_pos, next_pos,
                                                        control_intensity, self.rng, num_points=15)

                    # Draw curve as series of circles
                    for j, point in enumerate(curve_points):
                        t = j / len(curve_points)
                        point_width = segment_width * (1 - t * 0.3)
                        alpha = 0.7 - seg * 0.1

                        circle = Circle(point, point_width, color=arm_color, alpha=max(alpha, 0.3))
                        elements.append(circle)

                # Add branching
                if seg > 1 and params['branching'] > 0.5 and self.rng.random() < 0.3:
                    branch_dir = np.array([-current_dir[1], current_dir[0]])
                    if self.rng.random() < 0.5:
                        branch_dir = -branch_dir

                    branch_length = segment_length * 0.6
                    branch_end = current_pos + branch_dir * branch_length

                    # Lighter color for branches
                    branch_hsv = [arm_hsv[0], arm_hsv[1] * 0.7, arm_hsv[2] * 1.1]
                    branch_color = hsv_to_rgb([max(0, min(1, c)) for c in branch_hsv])

                    branch_curve = generate_organic_curve(current_pos, branch_end,
                                                        control_intensity * 0.5, self.rng, 12)
                    for point in branch_curve:
                        circle = Circle(point, segment_width * 0.4, color=branch_color, alpha=0.5)
                        elements.append(circle)

                current_pos = next_pos

            # Arm tip decoration
            tip_size = 0.05 + params['tip_decoration'] * 0.15
            tip_circle = Circle(current_pos, tip_size, color=arm_color, alpha=1.0)
            elements.append(tip_circle)

        return elements

    def generate_bilateral_biomorph(self, center, params, base_hsv, subtract_hsv):
        """Generate bilaterally symmetric biomorph"""
        elements = []

        # Main body as organic ellipse
        body_length = 1.5 + params['body_size'] * 2.0
        body_width = 0.8 + params['body_width'] * 1.2
        body_rotation = params['body_rotation'] * 360

        # Create segmented body
        num_body_segments = 3 + int(params['segmentation'] * 5)
        segment_length = body_length / num_body_segments

        for i in range(num_body_segments):
            # Segment position
            x_offset = (i - num_body_segments/2) * segment_length
            segment_center = [center[0] + x_offset, center[1]]

            # Segment size (wider in middle)
            middle_factor = 1.0 - abs(i - num_body_segments/2) / (num_body_segments/2)
            segment_width = body_width * (0.3 + middle_factor * 0.7)

            # Segment color
            color_factor = i / num_body_segments
            segment_hsv = [
                wrap_hue(base_hsv[0] - subtract_hsv[0] * color_factor * 0.3),
                max(0, min(1, base_hsv[1] - subtract_hsv[1] * color_factor * 0.2)),
                max(0, min(1, base_hsv[2] - subtract_hsv[2] * color_factor * 0.1))
            ]
            segment_color = hsv_to_rgb(segment_hsv)

            ellipse = Ellipse(segment_center, segment_length * 1.2, segment_width,
                            angle=body_rotation, color=segment_color, alpha=0.7)
            elements.append(ellipse)

        # Bilateral appendages
        num_appendage_pairs = 2 + int(params['arm_count'] * 4)

        for i in range(num_appendage_pairs):
            # Check activation
            arm_idx = i % len(params['arm_activations'])
            if params['arm_activations'][arm_idx] < 0.6:
                continue

            # Position along body
            body_t = (i + 1) / (num_appendage_pairs + 1)
            attach_x = center[0] + (body_t - 0.5) * body_length * 0.8

            # Appendage properties
            app_length = 0.8 + params['arm_length'] * 1.5
            app_angle = 30 + params['arm_angle'] * 60  # 30-90 degrees

            # Color for this appendage pair
            color_factor = i / num_appendage_pairs
            app_hsv = [
                wrap_hue(base_hsv[0] - subtract_hsv[0] * color_factor),
                max(0, min(1, base_hsv[1] - subtract_hsv[1] * color_factor * 0.5)),
                max(0, min(1, base_hsv[2] - subtract_hsv[2] * color_factor * 0.3))
            ]
            app_color = hsv_to_rgb(app_hsv)

            # Generate both sides
            for side in [-1, 1]:
                attach_y = center[1] + side * body_width * 0.4
                attach_point = np.array([attach_x, attach_y])

                # Appendage direction
                app_dir = np.array([np.cos(np.deg2rad(app_angle * side)),
                                  np.sin(np.deg2rad(app_angle * side))])
                app_end = attach_point + app_dir * app_length

                # Create organic appendage
                control_intensity = params['curvature'] * 0.4
                app_curve = generate_organic_curve(attach_point, app_end,
                                                 control_intensity, self.rng, 20)

                # Draw appendage with tapering width
                for j, point in enumerate(app_curve):
                    t = j / len(app_curve)
                    width = (0.08 + params['arm_width'] * 0.12) * (1 - t * 0.6)
                    alpha = 0.8 - t * 0.3

                    circle = Circle(point, width, color=app_color, alpha=alpha)
                    elements.append(circle)

                # Add joints
                if params['segmentation'] > 0.5:
                    joint_positions = [0.3, 0.6, 0.9]
                    for joint_t in joint_positions:
                        if joint_t < len(app_curve) / len(app_curve):
                            joint_idx = int(joint_t * len(app_curve))
                            joint_pos = app_curve[joint_idx]
                            joint_size = 0.06 + params['joint_size'] * 0.08

                            joint_circle = Circle(joint_pos, joint_size,
                                                color=app_color, alpha=1.0,
                                                linewidth=1, edgecolor='white')
                            elements.append(joint_circle)

        return elements

    def generate_spiral_biomorph(self, center, params, base_hsv, subtract_hsv):
        """Generate spiral-based biomorph"""
        elements = []

        # Spiral parameters
        num_turns = 2 + params['spiral_turns'] * 4
        max_radius = 1.5 + params['spiral_size'] * 2.0
        spiral_tightness = 0.8 + params['spiral_tightness'] * 1.2

        # Generate main spiral
        num_points = int(50 + params['detail_level'] * 100)
        t_values = np.linspace(0, num_turns * 2 * np.pi, num_points)

        spiral_points = []
        spiral_colors = []

        for i, t in enumerate(t_values):
            # Spiral position
            radius = (t / (num_turns * 2 * np.pi)) * max_radius
            x = center[0] + radius * np.cos(t * spiral_tightness)
            y = center[1] + radius * np.sin(t * spiral_tightness)

            # Add organic noise
            noise_factor = params['irregularity'] * 0.3
            noise = self.rng.normal(0, noise_factor, 2)
            spiral_points.append([x + noise[0], y + noise[1]])

            # Color progression along spiral
            progress = i / len(t_values)
            spiral_hsv = [
                wrap_hue(base_hsv[0] + progress * subtract_hsv[0]),
                max(0, min(1, base_hsv[1] - progress * subtract_hsv[1] * 0.5)),
                max(0, min(1, base_hsv[2] + progress * subtract_hsv[2] * 0.3))
            ]
            spiral_colors.append(hsv_to_rgb(spiral_hsv))

        # Draw spiral as connected segments
        for i in range(len(spiral_points) - 1):
            progress = i / len(spiral_points)
            width = (0.15 + params['arm_width'] * 0.2) * (1 - progress * 0.7)

            # Create segment
            start_point = spiral_points[i]
            end_point = spiral_points[i + 1]

            # Organic curve between points
            if i % 3 == 0:  # Add curves every few segments
                control_intensity = params['curvature'] * 0.2
                segment_curve = generate_organic_curve(np.array(start_point),
                                                     np.array(end_point),
                                                     control_intensity, self.rng, 8)
                for point in segment_curve:
                    circle = Circle(point, width, color=spiral_colors[i], alpha=0.7)
                    elements.append(circle)
            else:
                # Simple circle
                circle = Circle(start_point, width, color=spiral_colors[i], alpha=0.7)
                elements.append(circle)

        # Add spiral branches
        branch_frequency = max(5, int(20 - params['branching'] * 15))
        for i in range(0, len(spiral_points), branch_frequency):
            if i < len(spiral_points) - 1 and params['branching'] > 0.3:
                base_point = spiral_points[i]

                # Branch direction (perpendicular to spiral)
                if i < len(spiral_points) - 1:
                    spiral_dir = np.array(spiral_points[i+1]) - np.array(spiral_points[i])
                    spiral_dir = spiral_dir / (np.linalg.norm(spiral_dir) + 1e-8)

                    branch_dir = np.array([-spiral_dir[1], spiral_dir[0]])
                    if self.rng.random() < 0.5:
                        branch_dir = -branch_dir

                    branch_length = 0.3 + params['branch_length'] * 0.8
                    branch_end = np.array(base_point) + branch_dir * branch_length

                    # Branch color
                    branch_hsv = [
                        wrap_hue(base_hsv[0] + subtract_hsv[0] * 0.3),
                        max(0, min(1, base_hsv[1] + subtract_hsv[1] * 0.2)),
                        max(0, min(1, base_hsv[2] - subtract_hsv[2] * 0.2))
                    ]
                    branch_color = hsv_to_rgb(branch_hsv)

                    # Create organic branch
                    branch_curve = generate_organic_curve(np.array(base_point), branch_end,
                                                        params['curvature'] * 0.3,
                                                        self.rng, 12)

                    for j, point in enumerate(branch_curve):
                        t = j / len(branch_curve)
                        branch_width = 0.08 * (1 - t * 0.8)
                        alpha = 0.6 - t * 0.2

                        circle = Circle(point, branch_width, color=branch_color, alpha=alpha)
                        elements.append(circle)

        return elements

def create_biomorph(params):
    """
    Create an organic biomorphic creature from 32 parameters.

    Parameters structure:
    0-9: Arm/appendage activations
    10-12: Base HSV color
    13-15: Subtract HSV color
    16: Symmetry type (0=radial, 0.5=bilateral, 1=spiral)
    17: Symmetry strength
    18-19: Body parameters (size, complexity)
    20-22: Arm parameters (length, width, count)
    23-25: Growth parameters (curvature, branching, segmentation)
    26-28: Detail parameters (irregularity, decoration, joints)
    29-31: Special parameters (spiral properties, etc.)
    """
    if len(params) != 32:
        raise ValueError("Must provide exactly 32 parameters")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('black')

    # Extract parameter groups
    arm_activations = params[0:10]
    base_hsv = params[10:13]
    subtract_hsv = params[13:16]
    symmetry_type = params[16]
    symmetry_strength = params[17]

    # Create parameter dictionary for biomorph generation
    biomorph_params = {
        'arm_activations': arm_activations,
        'body_size': params[18],
        'body_complexity': params[19],
        'arm_length': params[20],
        'arm_width': params[21],
        'arm_count': params[22],
        'curvature': params[23],
        'branching': params[24],
        'segmentation': params[25],
        'irregularity': params[26],
        'tip_decoration': params[27],
        'joint_size': params[28],
        'spiral_turns': params[29],
        'spiral_size': params[29],
        'spiral_tightness': params[30],
        'detail_level': params[26],
        'branch_length': params[24],
        'growth_variation': params[31],
        'body_width': params[19],
        'body_rotation': params[25],
        'arm_angle': params[23]
    }

    # Create biomorph generator
    seed = int(params[28] * 9999) + 1
    generator = OrganicBiomorph(seed)
    center = np.array([0, 0])

    # Generate biomorph based on symmetry type
    if symmetry_type < 0.4:  # Radial
        elements = generator.generate_radial_biomorph(center, biomorph_params, base_hsv, subtract_hsv)
        morph_type = "Radial"
    elif symmetry_type < 0.7:  # Bilateral
        elements = generator.generate_bilateral_biomorph(center, biomorph_params, base_hsv, subtract_hsv)
        morph_type = "Bilateral"
    else:  # Spiral
        elements = generator.generate_spiral_biomorph(center, biomorph_params, base_hsv, subtract_hsv)
        morph_type = "Spiral"

    # Apply symmetry transformations to anchor points if needed
    if symmetry_strength > 0.1 and symmetry_type < 0.7:
        # Extract points from elements for symmetry application
        element_centers = []
        for element in elements:
            if hasattr(element, 'center'):
                element_centers.append(list(element.center))

        if element_centers:
            symmetric_points = apply_symmetry_organic(element_centers, symmetry_type,
                                                    symmetry_strength, center)

            # Update element positions (simplified approach)
            for i, element in enumerate(elements[:len(symmetric_points)]):
                if hasattr(element, 'center'):
                    element.center = symmetric_points[i]

    # Add all elements to the plot
    for element in elements:
        ax.add_patch(element)

    plt.title(f'Organic Biomorph - {morph_type} (Symmetry: {symmetry_strength:.2f})',
              color='white', fontsize=14)
    plt.tight_layout()
    return fig

def randomize_biomorph_parameters():
    """Generate random parameters for the organic biomorph."""
    return np.random.random(32)

def display_biomorph_gallery(rows=2, cols=3):
    """Display a gallery of different biomorphs."""
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    fig.patch.set_facecolor('black')

    # Create different types of biomorphs
    symmetry_types = [0.1, 0.3, 0.5, 0.6, 0.8, 0.9]

    for i, ax in enumerate(axes.flat):
        # Generate parameters with specific symmetry type
        params = randomize_biomorph_parameters()
        params[16] = symmetry_types[i % len(symmetry_types)]  # Symmetry type
        params[17] = 0.4 + (i % 3) * 0.3  # Varying symmetry strength

        # Create simplified version for subplot
        create_biomorph_subplot(ax, params, i)

    plt.suptitle('Organic Biomorph Gallery - Natural Forms',
                 color='white', fontsize=16, y=0.95)
    plt.tight_layout()
    return fig

def create_biomorph_subplot(ax, params, index):
    """Create a simplified biomorph for subplot display."""
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('black')

    # Simplified rendering for gallery view
    seed = int(params[28] * 9999) + 1
    generator = OrganicBiomorph(seed)
    center = np.array([0, 0])

    # Extract key parameters
    base_hsv = params[10:13]
    subtract_hsv = params[13:16]
    symmetry_type = params[16]

    # Create simplified parameter dict
    simple_params = {
        'arm_activations': params[0:10],
        'body_size': params[18] * 0.7,  # Scale down for subplot
        'body_complexity': params[19],
        'arm_length': params[20] * 0.8,
        'arm_width': params[21],
        'arm_count': params[22],
        'curvature': params[23],
        'branching': params[24],
        'segmentation': params[25],
        'irregularity': params[26],
        'tip_decoration': params[27],
        'joint_size': params[28],
        'spiral_turns': params[29],
        'spiral_size': params[29] * 0.7,
        'spiral_tightness': params[30],
        'detail_level': params[26],
        'branch_length': params[24],
        'growth_variation': params[31],
        'body_width': params[19],
        'body_rotation': params[25],
        'arm_angle': params[23]
    }

    # Generate appropriate biomorph type
    if symmetry_type < 0.4:
        elements = generator.generate_radial_biomorph(center, simple_params, base_hsv, subtract_hsv)
        title = f"Radial {index+1}"
    elif symmetry_type < 0.7:
        elements = generator.generate_bilateral_biomorph(center, simple_params, base_hsv, subtract_hsv)
        title = f"Bilateral {index+1}"
    else:
        elements = generator.generate_spiral_biomorph(center, simple_params, base_hsv, subtract_hsv)
        title = f"Spiral {index+1}"

    # Add elements to subplot
    for element in elements:
        ax.add_patch(element)

    ax.set_title(title, color='white', fontsize=10)

# Example usage
if __name__ == "__main__":
    # Create a single organic biomorph
    random_params = randomize_biomorph_parameters()
    print("Random parameters (32):", [f"{p:.2f}" for p in random_params])

    fig1 = create_biomorph(random_params)
    plt.show()

    # Show gallery of different biomorphs
    fig2 = display_biomorph_gallery()
    plt.show()