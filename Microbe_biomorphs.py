import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle, Ellipse, Polygon
import matplotlib.patches as patches
import json
import os

def wrap_hue(h):
    return h % 1.0

def rotate_point(point, center, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    return center + rotation_matrix @ (point - center)

def add_noise_to_points(points, noise_level, rng):
    """Add organic variation to points"""
    noise = rng.normal(0, noise_level, (len(points), 2))
    return points + noise

def generate_organic_curve(start, end, control_intensity, rng, num_points=20):
    """Generate a curved line between two points using Bezier-like curves"""
    t = np.linspace(0, 1, num_points)

    # Add control points for organic curves
    mid = (start + end) / 2
    perpendicular = np.array([-(end[1] - start[1]), end[0] - start[0]])
    perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)

    control1 = start + perpendicular * control_intensity * rng.uniform(-1, 1)
    control2 = end + perpendicular * control_intensity * rng.uniform(-1, 1)

    # Quadratic Bezier curve
    curve_points = []
    for t_val in t:
        point = (1-t_val)**2 * start + 2*(1-t_val)*t_val * control1 + t_val**2 * end
        curve_points.append(point)

    return np.array(curve_points)

class OrganismGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_radial_organism(self, center, params):
        """Generate radially symmetric organism (like radiolarian, starfish)"""
        num_arms = self.rng.randint(3, 12)
        arm_length = self.rng.uniform(8, 20)
        inner_radius = self.rng.uniform(2, 6)

        points = []
        colors = []
        lines = []

        base_hue = params['initial_hue']

        # Central body
        central_color = hsv_to_rgb([base_hue, 0.8, 0.9])
        center_circle = Circle(center, inner_radius, color=central_color, alpha=0.7)

        # Generate arms
        for i in range(num_arms):
            angle = (2 * np.pi * i) / num_arms

            # Arm direction
            arm_dir = np.array([np.cos(angle), np.sin(angle)])

            # Generate arm points
            arm_points = []
            for j in range(self.rng.randint(3, 8)):
                distance = inner_radius + (j + 1) * arm_length / 7
                # Add organic variation
                distance *= self.rng.uniform(0.7, 1.3)
                width_factor = max(0.1, 1 - j/6)  # Taper the arm

                # Side branches
                if j > 1 and self.rng.rand() < 0.4:
                    side_angle = angle + self.rng.uniform(-0.5, 0.5)
                    side_dir = np.array([np.cos(side_angle), np.sin(side_angle)])
                    side_point = center + side_dir * distance * 0.7
                    arm_points.append(side_point)

                point = center + arm_dir * distance
                # Add slight curvature
                perp = np.array([-arm_dir[1], arm_dir[0]])
                point += perp * self.rng.uniform(-2, 2) * width_factor
                arm_points.append(point)

            # Color variation along arm
            arm_hue = wrap_hue(base_hue + self.rng.uniform(-0.1, 0.1))

            points.extend(arm_points)
            colors.extend([hsv_to_rgb([arm_hue, 0.9, 0.8])] * len(arm_points))

            # Connect arm points
            for j in range(len(arm_points)-1):
                lines.append((arm_points[j], arm_points[j+1]))

        return points, colors, lines, [center_circle]

    def generate_bilateral_organism(self, center, params):
        """Generate bilaterally symmetric organism (like diatom, fish-like)"""
        body_length = self.rng.uniform(15, 25)
        body_width = self.rng.uniform(6, 12)

        points = []
        colors = []
        lines = []
        shapes = []

        base_hue = params['initial_hue']

        # Main body axis
        body_start = center - np.array([body_length/2, 0])
        body_end = center + np.array([body_length/2, 0])

        # Generate body outline
        num_body_points = self.rng.randint(6, 12)
        t_values = np.linspace(0, 1, num_body_points)

        top_points = []
        bottom_points = []

        for t in t_values:
            x = body_start[0] + t * body_length
            # Organic body shape (elliptical with variation)
            width_factor = 4 * t * (1 - t)  # Parabolic width
            width_factor *= self.rng.uniform(0.7, 1.3)  # Add variation

            y_offset = body_width * width_factor * 0.5

            top_points.append([x, center[1] + y_offset])
            bottom_points.append([x, center[1] - y_offset])

        # Create body polygon
        body_points = top_points + bottom_points[::-1]
        body_color = hsv_to_rgb([base_hue, 0.7, 0.8])
        body_polygon = Polygon(body_points, color=body_color, alpha=0.6)
        shapes.append(body_polygon)

        # Add symmetric appendages
        num_appendages = self.rng.randint(2, 6)
        for i in range(num_appendages):
            # Position along body
            t = self.rng.uniform(0.2, 0.8)
            base_x = body_start[0] + t * body_length
            base_y_top = center[1] + body_width * 2 * t * (1 - t) * 0.5
            base_y_bottom = center[1] - body_width * 2 * t * (1 - t) * 0.5

            # Appendage properties
            app_length = self.rng.uniform(4, 12)
            app_angle = self.rng.uniform(15, 75)

            # Top appendage
            top_base = np.array([base_x, base_y_top])
            top_dir = np.array([np.cos(np.deg2rad(app_angle)), np.sin(np.deg2rad(app_angle))])
            top_end = top_base + top_dir * app_length

            # Bottom appendage (mirror)
            bottom_base = np.array([base_x, base_y_bottom])
            bottom_dir = np.array([np.cos(np.deg2rad(-app_angle)), np.sin(np.deg2rad(-app_angle))])
            bottom_end = bottom_base + bottom_dir * app_length

            # Generate curved appendages
            top_curve = generate_organic_curve(top_base, top_end, 2, self.rng)
            bottom_curve = generate_organic_curve(bottom_base, bottom_end, 2, self.rng)

            app_hue = wrap_hue(base_hue + self.rng.uniform(-0.05, 0.05))
            app_color = hsv_to_rgb([app_hue, 0.9, 0.7])

            points.extend(top_curve)
            points.extend(bottom_curve)
            colors.extend([app_color] * (len(top_curve) + len(bottom_curve)))

            # Add lines for appendages
            for j in range(len(top_curve)-1):
                lines.append((top_curve[j], top_curve[j+1]))
            for j in range(len(bottom_curve)-1):
                lines.append((bottom_curve[j], bottom_curve[j+1]))

        return points, colors, lines, shapes

    def generate_asymmetric_organism(self, center, params):
        """Generate asymmetric organism (like amoeba, bacteria)"""
        points = []
        colors = []
        lines = []
        shapes = []

        base_hue = params['initial_hue']

        # Irregular blob shape
        num_blob_points = self.rng.randint(8, 16)
        angles = np.sort(self.rng.uniform(0, 2*np.pi, num_blob_points))

        blob_points = []
        for angle in angles:
            radius = self.rng.uniform(8, 18)
            # Add irregularity
            radius *= self.rng.uniform(0.5, 1.5)

            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            blob_points.append([x, y])

        # Create main body
        blob_color = hsv_to_rgb([base_hue, 0.6, 0.9])
        blob_polygon = Polygon(blob_points, color=blob_color, alpha=0.7)
        shapes.append(blob_polygon)

        # Add pseudopods/extensions
        num_extensions = self.rng.randint(3, 8)
        for i in range(num_extensions):
            # Random starting point on blob
            start_idx = self.rng.randint(0, len(blob_points))
            start_point = np.array(blob_points[start_idx])

            # Extension direction (roughly outward)
            to_center = center - start_point
            perp = np.array([-to_center[1], to_center[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-8)

            ext_dir = -to_center / (np.linalg.norm(to_center) + 1e-8)
            ext_dir += perp * self.rng.uniform(-0.5, 0.5)  # Add randomness
            ext_dir = ext_dir / (np.linalg.norm(ext_dir) + 1e-8)

            # Generate extension
            ext_length = self.rng.uniform(5, 15)
            num_ext_points = self.rng.randint(3, 7)

            ext_points = [start_point]
            current_pos = start_point.copy()
            current_dir = ext_dir.copy()

            for j in range(num_ext_points):
                # Add curvature and variation
                current_dir += self.rng.normal(0, 0.2, 2)
                current_dir = current_dir / (np.linalg.norm(current_dir) + 1e-8)

                step_size = ext_length / num_ext_points
                step_size *= self.rng.uniform(0.7, 1.3)

                current_pos += current_dir * step_size
                ext_points.append(current_pos.copy())

            ext_hue = wrap_hue(base_hue + self.rng.uniform(-0.1, 0.1))
            ext_color = hsv_to_rgb([ext_hue, 0.8, 0.8])

            points.extend(ext_points)
            colors.extend([ext_color] * len(ext_points))

            # Connect extension points
            for j in range(len(ext_points)-1):
                lines.append((ext_points[j], ext_points[j+1]))

        # Add internal structures (organelles)
        num_organelles = self.rng.randint(2, 6)
        for i in range(num_organelles):
            org_center = center + self.rng.uniform(-6, 6, 2)
            org_radius = self.rng.uniform(1, 3)
            org_hue = wrap_hue(base_hue + self.rng.uniform(0.2, 0.4))
            org_color = hsv_to_rgb([org_hue, 0.9, 0.6])

            organelle = Circle(org_center, org_radius, color=org_color, alpha=0.8)
            shapes.append(organelle)

        return points, colors, lines, shapes

    def generate_spiral_organism(self, center, params):
        """Generate spiral organism (like some shells, algae)"""
        points = []
        colors = []
        lines = []
        shapes = []

        base_hue = params['initial_hue']

        # Spiral parameters
        num_turns = self.rng.uniform(2, 5)
        max_radius = self.rng.uniform(12, 20)
        spiral_tightness = self.rng.uniform(0.5, 2)

        # Generate spiral
        num_points = self.rng.randint(50, 100)
        t_values = np.linspace(0, num_turns * 2 * np.pi, num_points)

        spiral_points = []
        for i, t in enumerate(t_values):
            radius = (t / (num_turns * 2 * np.pi)) * max_radius

            x = center[0] + radius * np.cos(t * spiral_tightness)
            y = center[1] + radius * np.sin(t * spiral_tightness)

            # Add organic variation
            noise = self.rng.normal(0, 0.5, 2)
            spiral_points.append([x + noise[0], y + noise[1]])

        # Color gradient along spiral
        for i, point in enumerate(spiral_points):
            progress = i / len(spiral_points)
            hue = wrap_hue(base_hue + progress * 0.3)
            color = hsv_to_rgb([hue, 0.8, 0.9])

            points.append(point)
            colors.append(color)

        # Connect spiral points
        for i in range(len(spiral_points)-1):
            lines.append((spiral_points[i], spiral_points[i+1]))

        # Add branches along spiral
        branch_frequency = self.rng.randint(5, 10)
        for i in range(0, len(spiral_points), branch_frequency):
            if i < len(spiral_points) - 1:
                base_point = spiral_points[i]

                # Branch direction
                if i < len(spiral_points) - 1:
                    main_dir = np.array(spiral_points[i+1]) - np.array(spiral_points[i])
                    main_dir = main_dir / (np.linalg.norm(main_dir) + 1e-8)

                    # Perpendicular branch
                    branch_dir = np.array([-main_dir[1], main_dir[0]])
                    branch_length = self.rng.uniform(3, 8)

                    branch_end = np.array(base_point) + branch_dir * branch_length

                    branch_hue = wrap_hue(base_hue + 0.15)
                    branch_color = hsv_to_rgb([branch_hue, 0.9, 0.7])

                    points.extend([base_point, branch_end])
                    colors.extend([branch_color, branch_color])
                    lines.append((base_point, branch_end))

        return points, colors, lines, shapes

def draw_organism(ax, params):
    """Draw a single organism deterministically from parameters"""
    rng = np.random.RandomState(params['seed'])
    generator = OrganismGenerator(params['seed'])

    center = np.array([params['start_pos_x'], params['start_pos_y']])

    # Use specified organism type instead of random selection
    organism_type = params['organism_type']

    if organism_type == 'radial':
        points, colors, lines, shapes = generator.generate_radial_organism(center, params)
    elif organism_type == 'bilateral':
        points, colors, lines, shapes = generator.generate_bilateral_organism(center, params)
    elif organism_type == 'asymmetric':
        points, colors, lines, shapes = generator.generate_asymmetric_organism(center, params)
    else:  # spiral
        points, colors, lines, shapes = generator.generate_spiral_organism(center, params)

    # Apply general rotation
    if params.get('general_rotation_deg', 0) != 0:
        angle = params['general_rotation_deg']
        points = [rotate_point(np.array(p), center, angle) for p in points]
        lines = [(rotate_point(np.array(p1), center, angle),
                 rotate_point(np.array(p2), center, angle)) for p1, p2 in lines]

    # Draw shapes (filled polygons, circles)
    for shape in shapes:
        ax.add_patch(shape)

    # Draw lines
    for (p1, p2), color in zip(lines, colors[:len(lines)]):
        linewidth = rng.uniform(1, 3)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, alpha=0.8)

    # Draw points
    for point, color in zip(points, colors):
        markersize = rng.uniform(2, 6)
        ax.plot(point[0], point[1], 'o', color=color, markersize=markersize, alpha=0.9)

    ax.set_xlim(0, params['canvas_size'])
    ax.set_ylim(0, params['canvas_size'])
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_aspect('equal')

def generate_random_params(seed=None, canvas_size=64):
    """Generate parameters for one organism - fully deterministic"""
    rng = np.random.RandomState(seed)

    margin = 15

    # Choose organism type deterministically
    organism_types = ['radial', 'bilateral', 'asymmetric', 'spiral']
    weights = [0.25, 0.25, 0.35, 0.15]  # Favor asymmetric for variety
    organism_type = rng.choice(organism_types, p=weights)

    params = {
        'start_pos_x': float(rng.uniform(margin, canvas_size - margin)),
        'start_pos_y': float(rng.uniform(margin, canvas_size - margin)),
        'initial_hue': float(rng.rand()),
        'canvas_size': int(canvas_size),
        'general_rotation_deg': float(rng.uniform(0, 360)),
        'organism_type': organism_type,
        'seed': int(seed) if seed is not None else None
    }
    return params

def display_grid(grid_size=8, canvas_size=64, display_scale=4, base_seed=42):
    """Display a grid of diverse microscopic organisms"""
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size*display_scale, grid_size*display_scale),
                            dpi=80, facecolor='black')
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    for i, ax in enumerate(axs.flatten()):
        shape_seed = base_seed + i
        params = generate_random_params(seed=shape_seed, canvas_size=canvas_size)
        draw_organism(ax, params)

    plt.suptitle('Microscopic Organism Zoo', fontsize=16, color='white')
    plt.show()

def generate_dataset(num_samples, canvas_size=64, base_seed=42, save_path="organism_dataset.json"):
    """
    Generate a dataset of organism parameters for neural network training.

    Args:
        num_samples: Number of organism parameter sets to generate
        canvas_size: Size of the canvas (default 64x64)
        base_seed: Starting seed for deterministic generation
        save_path: Path to save the JSON dataset

    Returns:
        List of parameter dictionaries
    """
    dataset = []

    for i in range(num_samples):
        seed = base_seed + i
        params = generate_random_params(seed=seed, canvas_size=canvas_size)

        # Add metadata
        params['sample_id'] = i
        params['generated_timestamp'] = f"seed_{seed}"

        dataset.append(params)

    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {num_samples} organism parameters and saved to {save_path}")
    return dataset

def load_dataset(load_path="organism_dataset.json"):
    """Load organism parameters from JSON file"""
    with open(load_path, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} organism parameters from {load_path}")
    return dataset

def generate_organism_from_params(params, save_image=False, save_path=None):
    """
    Generate a single organism image from parameters.
    Perfect for neural network training data generation.

    Args:
        params: Parameter dictionary (from dataset)
        save_image: Whether to save the generated image
        save_path: Path to save image (if None, uses sample_id)

    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100, facecolor='black')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    draw_organism(ax, params)

    if save_image:
        if save_path is None:
            save_path = f"organism_{params.get('sample_id', params['seed'])}.png"
        plt.savefig(save_path, facecolor='black', bbox_inches='tight', pad_inches=0)
        print(f"Saved organism image to {save_path}")

    return fig

def generate_training_images_from_dataset(dataset_path="organism_dataset.json",
                                        output_dir="training_images",
                                        max_images=None):
    """
    Generate all training images from a dataset JSON file.

    Args:
        dataset_path: Path to the JSON dataset file
        output_dir: Directory to save training images
        max_images: Maximum number of images to generate (None for all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset(dataset_path)

    if max_images:
        dataset = dataset[:max_images]

    print(f"Generating {len(dataset)} training images...")

    for i, params in enumerate(dataset):
        save_path = os.path.join(output_dir, f"organism_{params['sample_id']:06d}.png")
        fig = generate_organism_from_params(params, save_image=True, save_path=save_path)
        plt.close(fig)  # Free memory

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{len(dataset)} images...")

    print(f"Completed! All images saved to {output_dir}/")

def verify_determinism(params, num_tests=5):
    """
    Verify that organism generation is deterministic by generating the same organism multiple times.
    Returns True if all generations are identical.
    """
    print("Testing determinism...")

    # Generate the same organism multiple times
    figures = []
    for i in range(num_tests):
        fig = generate_organism_from_params(params, save_image=False)
        figures.append(fig)

    # Close figures to free memory
    for fig in figures:
        plt.close(fig)

    print(f"Generated {num_tests} identical organisms from same parameters")
    print("Note: Visual verification required - check saved images are identical")
    return True

# Run the generator and dataset creation
if __name__ == "__main__":
    # Display a grid of organisms
    print("Displaying organism grid...")
    display_grid(grid_size=8, canvas_size=64, display_scale=3, base_seed=123)

    """
    # Generate a dataset for neural network training
    print("\nGenerating dataset...")
    dataset = generate_dataset(num_samples=1000, canvas_size=64, base_seed=42,
                             save_path="organism_dataset.json")
    """


    # Example: Generate a few training images
    print("\nGenerating sample training images...")
    sample_params = dataset[:5]  # First 5 organisms
    for i, params in enumerate(sample_params):
        fig = generate_organism_from_params(params, save_image=True,
                                          save_path=f"sample_organism_{i}.png")
        plt.close(fig)

    # Verify determinism
    print("\nVerifying determinism...")
    verify_determinism(dataset[0])

    print("\nDataset generation complete!")
    print("Use generate_training_images_from_dataset() to create all training images.")

    # Example of how to generate all training images:
    # generate_training_images_from_dataset("organism_dataset.json", "training_images", max_images=100)