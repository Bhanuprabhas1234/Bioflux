import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from scipy import ndimage
import tifffile
import matplotlib
import os
from scipy.ndimage import gaussian_filter

class NDVIPredatorPreySimulation:
    def __init__(self, tiff_folder_path, num_herbivores=1000, num_carnivores=200,
                 ndvi_update_frequency=0.001, rl_herbivores=False):
        # Initialize parameters
        self.num_herbivores = num_herbivores
        self.num_carnivores = num_carnivores
        self.ndvi_update_frequency = ndvi_update_frequency
        self.current_ndvi_index = 0
        self.rl_herbivores = rl_herbivores

        # Load NDVI data from TIFF files in the folder
        self.ndvi_files = sorted([os.path.join(tiff_folder_path, f) for f in os.listdir(tiff_folder_path) if f.endswith(".tiff")])
        if not self.ndvi_files:
            raise ValueError(f"No TIFF files found in the specified folder: {tiff_folder_path}")
        self.ndvi_grids = []
        self.original_ndvi_data_list = []
        for file_path in self.ndvi_files:
            normalized_array, original_array = self.load_ndvi_from_tiff(file_path)
            self.ndvi_grids.append(normalized_array)
            self.original_ndvi_data_list.append(original_array)

        # Determine grid size from the first NDVI file
        if self.ndvi_grids:
            first_ndvi_shape = self.ndvi_grids[0].shape
            self.grid_size = first_ndvi_shape[0]  # Assuming square or taking the first dimension
            if first_ndvi_shape[0] != first_ndvi_shape[1]:
                print(f"Warning: NDVI data is not square. Using dimensions {first_ndvi_shape}.")
        else:
            raise ValueError("No NDVI data loaded.")

        # Create grids matching the NDVI grid size
        self.herbivore_grid = np.zeros((self.grid_size, self.grid_size))
        self.carnivore_grid = np.zeros((self.grid_size, self.grid_size))

        # Track individual agents
        self.herbivores = []
        self.carnivores = []

        # Parameters
        self.herbivore_energy_gain = 5  # Energy gained from consuming vegetation
        self.herbivore_energy_loss = 1  # Energy lost per step
        self.herbivore_reproduce_threshold = 20  # Energy needed to reproduce
        self.herbivore_initial_energy = 10

        self.carnivore_energy_gain = 20  # Energy gained from consuming herbivores
        self.carnivore_energy_loss = 1  # Energy lost per step
        self.carnivore_reproduce_threshold = 30  # Energy needed to reproduce
        self.carnivore_initial_energy = 15

        # Statistics tracking
        self.herbivore_count_history = []
        self.carnivore_count_history = []
        self.ndvi_mean_history = []

        # Initialize the environment with the first NDVI data
        self.ndvi_grid = self.ndvi_grids[0]
        self.initialize_environment()

        # RL related attributes
        if self.rl_herbivores:
            self.q_table = {} # For Q-learning

    def load_ndvi_from_tiff(self, file_path):
        """
        Load NDVI data from TIFF file using precise normalization

        Parameters:
        file_path (str): Path to the TIFF file containing NDVI data
        """
        # Load TIFF with full precision
        pixel_array = tifffile.imread(file_path)

        # Calculate vmin and vmax excluding values <= -999
        if np.any(pixel_array > -999):
            valid_pixels = pixel_array[pixel_array > -999]
            vmin = np.percentile(valid_pixels, 2)
            vmax = np.percentile(valid_pixels, 98)

            # Create normalized version (0-1 scale based on vmin/vmax)
            normalized_array = np.clip((pixel_array - vmin) / (vmax - vmin), 0, 1)

            return normalized_array, pixel_array
        else:
            raise ValueError(f"No valid NDVI data found in the file: {file_path}")

    def initialize_environment(self):
        """Initialize the simulation environment with NDVI data and animals"""
        # Place herbivores randomly
        for _ in range(self.num_herbivores):
            x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
            herbivore = {
                'x': x,
                'y': y,
                'energy': self.herbivore_initial_energy
            }
            if self.rl_herbivores:
                herbivore['q_table'] = {} # Each RL herbivore can have its own Q-table or share one
                herbivore['learning_rate'] = 0.1
                herbivore['discount_factor'] = 0.9
                herbivore['exploration_rate'] = 0.3
            self.herbivores.append(herbivore)
            self.herbivore_grid[x, y] += 1

        # Place carnivores randomly
        for _ in range(self.num_carnivores):
            x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
            self.carnivores.append({
                'x': x,
                'y': y,
                'energy': self.carnivore_initial_energy
            })
            self.carnivore_grid[x, y] += 1

    def update_ndvi(self, step):
        """Update NDVI based on the simulation step and available TIFF files"""
        if self.ndvi_files:
            ndvi_index = step // self.ndvi_update_frequency
            if 0 <= ndvi_index < len(self.ndvi_grids):
                self.ndvi_grid = self.ndvi_grids[ndvi_index]
                self.current_ndvi_index = ndvi_index
            elif self.current_ndvi_index < len(self.ndvi_grids) - 1:
                self.current_ndvi_index = len(self.ndvi_grids) - 1
                self.ndvi_grid = self.ndvi_grids[self.current_ndvi_index]

        # Apply logistic growth model to simulate vegetation regrowth
        growth_rate = 0.1  # Growth rate parameter
        carrying_capacity = 1.0  # Maximum NDVI value
        self.ndvi_grid += growth_rate * self.ndvi_grid * (carrying_capacity - self.ndvi_grid)
        self.ndvi_grid = np.clip(self.ndvi_grid, 0, carrying_capacity)  # Ensure NDVI stays within bounds

        # Apply diffusion to NDVI grid
        self.ndvi_grid = gaussian_filter(self.ndvi_grid, sigma=1)

    def diffuse_population(self, grid, sigma=1):
        """Apply diffusion to a population grid using a Gaussian kernel."""
        return gaussian_filter(grid, sigma=sigma)

    def get_herbivore_state(self, herbivore):
        """Define the state for the RL herbivore."""
        x, y = herbivore['x'], herbivore['y']
        # Example state: (x, y, discretized_ndvi)
        # You might want to discretize NDVI or energy for simpler state spaces
        ndvi_value = self.ndvi_grid[x, y]
        # Example of including nearby carnivore presence (optional)
        nearby_carnivores = np.sum(self.carnivore_grid[max(0, x-2):min(self.grid_size, x+3),
                                                     max(0, y-2):min(self.grid_size, y+3)]) > 0
        return (x, y, round(ndvi_value, 2), int(herbivore['energy'] > 5)) # Example state

    def choose_herbivore_action(self, herbivore, possible_actions):
        """Choose action for RL herbivore using epsilon-greedy policy."""
        state = self.get_herbivore_state(herbivore)
        if state not in herbivore['q_table']:
            herbivore['q_table'][state] = {action: 0 for action in possible_actions}

        if random.random() < herbivore['exploration_rate']:
            return random.choice(possible_actions)
        else:
            return max(herbivore['q_table'][state], key=herbivore['q_table'][state].get)

    def update_q_table(self, herbivore, old_state, action, reward, new_state):
        """Update Q-table for RL herbivore."""
        if new_state not in herbivore['q_table']:
            herbivore['q_table'][new_state] = {action: 0 for action in self.get_possible_herbivore_actions()}

        old_value = herbivore['q_table'][old_state][action]
        next_max = max(herbivore['q_table'][new_state].values())
        new_value = (1 - herbivore['learning_rate']) * old_value + \
                    herbivore['learning_rate'] * (reward + herbivore['discount_factor'] * next_max)
        herbivore['q_table'][old_state][action] = new_value

    def get_possible_herbivore_actions(self):
        """Define possible actions for herbivores."""
        return ['move_n', 'move_s', 'move_e', 'move_w', 'stay']

    def move_herbivores(self):
        """Move herbivores based on NDVI values or RL policy."""
        new_herbivore_grid = np.zeros((self.grid_size, self.grid_size))
        new_herbivores = []

        for herbivore in self.herbivores:
            old_x, old_y = herbivore['x'], herbivore['y']
            old_energy = herbivore['energy']
            old_state = None
            action = None

            # Lose energy per step
            herbivore['energy'] -= self.herbivore_energy_loss

            # Die if no energy
            if herbivore['energy'] <= 0:
                if self.rl_herbivores:
                    reward = -1 # Negative reward for dying
                    new_state = self.get_herbivore_state(herbivore) # State after death (can be same as before)
                    if old_state and action:
                        self.update_q_table(herbivore, old_state, action, reward, new_state)
                continue

            if self.rl_herbivores:
                possible_actions = self.get_possible_herbivore_actions()
                old_state = self.get_herbivore_state(herbivore)
                action = self.choose_herbivore_action(herbivore, possible_actions)

                new_x, new_y = old_x, old_y
                if action == 'move_n':
                    new_x = (old_x - 1) % self.grid_size
                elif action == 'move_s':
                    new_x = (old_x + 1) % self.grid_size
                elif action == 'move_e':
                    new_y = (old_y + 1) % self.grid_size
                elif action == 'move_w':
                    new_y = (old_y - 1) % self.grid_size
                # 'stay' action does not change coordinates

                # Calculate reward for the action
                reward = -0.01 # Small negative reward for each step

                # Check if moved to a higher NDVI cell
                current_ndvi = self.ndvi_grid[old_x, old_y]
                next_ndvi = self.ndvi_grid[new_x, new_y]
                if next_ndvi > current_ndvi:
                    reward += 0.1

                herbivore['x'], herbivore['y'] = new_x, new_y
            else:
                x, y = herbivore['x'], herbivore['y']
                # Find neighboring cells (Moore neighborhood)
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = (x + dx) % self.grid_size, (y + dy) % self.grid_size  # Wrap around
                        neighbors.append((nx, ny, self.ndvi_grid[nx, ny]))

                # Sort neighbors by NDVI value (herbivores prefer higher NDVI)
                neighbors.sort(key=lambda n: n[2], reverse=True)

                # Add some randomness to movement (80% choose best, 20% random)
                if random.random() < 0.8:
                    # Move to the cell with highest NDVI
                    move_to = neighbors[0]
                else:
                    # Random movement
                    move_to = random.choice(neighbors)

                new_x, new_y = move_to[0], move_to[1]
                herbivore['x'], herbivore['y'] = new_x, new_y

            # Consume vegetation (gain energy)
            energy_gain = self.ndvi_grid[herbivore['x'], herbivore['y']] * self.herbivore_energy_gain
            herbivore['energy'] += energy_gain
            if self.rl_herbivores:
                reward += energy_gain * 0.01 # Small positive reward for eating

            # Reduce NDVI value due to consumption (only if NDVI is positive)
            if self.ndvi_grid[herbivore['x'], herbivore['y']] > 0:
                self.ndvi_grid[herbivore['x'], herbivore['y']] = max(0, self.ndvi_grid[herbivore['x'], herbivore['y']] - 0.05) # Reduced consumption rate

            new_herbivore_grid[herbivore['x'], herbivore['y']] += 1

            # Reproduction
            if herbivore['energy'] > self.herbivore_reproduce_threshold:
                herbivore['energy'] /= 2  # Split energy
                offspring = {
                    'x': herbivore['x'],
                    'y': herbivore['y'],
                    'energy': herbivore['energy']
                }
                if self.rl_herbivores:
                    offspring['q_table'] = herbivore['q_table'].copy() # Inherit Q-table (optional)
                    offspring['learning_rate'] = herbivore['learning_rate']
                    offspring['discount_factor'] = herbivore['discount_factor']
                    offspring['exploration_rate'] = herbivore['exploration_rate']
                    if self.rl_herbivores:
                        reward += 0.5 # Reward for reproducing
                new_herbivores.append(offspring)
                new_herbivore_grid[herbivore['x'], herbivore['y']] += 1

            new_herbivores.append(herbivore)

            if self.rl_herbivores and old_state and action:
                new_state = self.get_herbivore_state(herbivore)
                self.update_q_table(herbivore, old_state, action, reward, new_state)

        self.herbivores = new_herbivores
        self.herbivore_grid = self.diffuse_population(new_herbivore_grid, sigma=1)

    def move_carnivores(self):
        """Move carnivores based on herbivore locations"""
        # (Carnivore movement logic remains the same as it's not an RL agent in this example)
        new_carnivore_grid = np.zeros((self.grid_size, self.grid_size))
        new_carnivores = []

        for carnivore in self.carnivores:
            # Lose energy per step
            carnivore['energy'] -= self.carnivore_energy_loss

            # Die if no energy
            if carnivore['energy'] <= 0:
                continue

            x, y = carnivore['x'], carnivore['y']

            # Find neighboring cells
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = (x + dx) % self.grid_size, (y + dy) % self.grid_size  # Wrap around
                    neighbors.append((nx, ny, self.herbivore_grid[nx, ny]))

            # Sort neighbors by herbivore count (carnivores prefer higher herbivore density)
            neighbors.sort(key=lambda n: n[2], reverse=True)

            # Add some randomness to movement (70% choose best, 30% random)
            if random.random() < 0.7:
                # Move to the cell with highest herbivore count
                move_to = neighbors[0]
            else:
                # Random movement
                move_to = random.choice(neighbors)

            new_x, new_y = move_to[0], move_to[1]

            # Hunt herbivores (gain energy)
            herbivores_here = self.herbivore_grid[new_x, new_y]
            if herbivores_here > 0:
                # Consume up to 2 herbivores maximum
                herbivores_eaten = min(2, herbivores_here)
                carnivore['energy'] += herbivores_eaten * self.carnivore_energy_gain

                # Remove eaten herbivores
                self.remove_herbivores(new_x, new_y, herbivores_eaten)

            # Update position
            carnivore['x'], carnivore['y'] = new_x, new_y
            new_carnivore_grid[new_x, new_y] += 1

            # Add to new list
            new_carnivores.append(carnivore)

            # Reproduction
            if carnivore['energy'] > self.carnivore_reproduce_threshold:
                carnivore['energy'] /= 2  # Split energy

                # Create offspring with same position
                offspring = {
                    'x': new_x,
                    'y': new_y,
                    'energy': carnivore['energy']
                }
                new_carnivores.append(offspring)
                new_carnivore_grid[new_x, new_y] += 1

        self.carnivores = new_carnivores
        self.carnivore_grid = self.diffuse_population(new_carnivore_grid, sigma=1)

    def remove_herbivores(self, x, y, count):
        """Remove herbivores that have been eaten by carnivores"""
        # Find herbivores at this location
        herbivores_here = [h for h in self.herbivores if h['x'] == x and h['y'] == y]

        # Remove them (up to count)
        removed = 0
        herbivores_to_remove = []
        for h in herbivores_here:
            if removed < count:
                herbivores_to_remove.append(h)
                removed += 1
            else:
                break

        for h in herbivores_to_remove:
            self.herbivores.remove(h)

        # Update herbivore grid
        self.herbivore_grid[x, y] = max(0, self.herbivore_grid[x, y] - count)

    def update(self, step):
        """Run one step of the simulation"""
        self.update_ndvi(step)
        self.move_herbivores()
        self.move_carnivores()

        # Update statistics
        self.herbivore_count_history.append(len(self.herbivores))
        self.carnivore_count_history.append(len(self.carnivores))
        self.ndvi_mean_history.append(np.mean(self.ndvi_grid))

    def run_simulation(self, steps=100, animate=True, save_animation=False):
        """Run the simulation for a specified number of steps"""
        if animate:
            fig = plt.figure(figsize=(10, 7))  # Increased figure size for better layout

            # Create subplots
            ax1 = plt.subplot2grid((2, 3), (0, 0))  # NDVI
            ax2 = plt.subplot2grid((2, 3), (0, 1))  # Herbivores
            ax3 = plt.subplot2grid((2, 3), (0, 2))  # Carnivores
            ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)  # Population history

            # NDVI colormap
            ndvi_cmap = plt.cm.YlGn

            # Initial plots
            ndvi_plot = ax1.imshow(self.ndvi_grid, cmap=ndvi_cmap, vmin=0, vmax=1)
            ax1.set_title('NDVI')
            plt.colorbar(ndvi_plot, ax=ax1, fraction=0.046, pad=0.04)

            herbivore_plot = ax2.imshow(self.herbivore_grid, cmap='Blues')
            ax2.set_title('Herbivores')
            plt.colorbar(herbivore_plot, ax=ax2, fraction=0.046, pad=0.04)

            carnivore_plot = ax3.imshow(self.carnivore_grid, cmap='Reds')
            ax3.set_title('Carnivores')
            plt.colorbar(carnivore_plot, ax=ax3, fraction=0.046, pad=0.04)

            line1, = ax4.plot([], [], 'g-', label='Herbivores')
            line2, = ax4.plot([], [], 'r-', label='Carnivores')
            line3, = ax4.plot([], [], 'b--', label='Mean NDVI x100')
            ax4.set_xlim(0, steps)
            ax4.set_ylim(0, max(self.num_herbivores * 2, self.num_carnivores * 5))  # Adjust as needed
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Count')
            ax4.legend()

            def init():
                ndvi_plot.set_data(self.ndvi_grid)
                herbivore_plot.set_data(self.herbivore_grid)
                carnivore_plot.set_data(self.carnivore_grid)
                line1.set_data([], [])
                line2.set_data([], [])
                line3.set_data([], [])
                return ndvi_plot, herbivore_plot, carnivore_plot, line1, line2, line3

            def animate(i):
                self.update(i)

                ndvi_plot.set_data(self.ndvi_grid)

                # Update herbivore plot and adjust colormap scale
                herbivore_plot.set_data(self.herbivore_grid)
                if np.max(self.herbivore_grid) > 0:
                    herbivore_plot.set_clim(0, max(1, np.max(self.herbivore_grid)))
                else:
                    herbivore_plot.set_clim(0, 1) # Ensure colormap is reset if no herbivores

                # Update carnivore plot and adjust colormap scale
                carnivore_plot.set_data(self.carnivore_grid)
                if np.max(self.carnivore_grid) > 0:
                    carnivore_plot.set_clim(0, max(1, np.max(self.carnivore_grid)))
                else:
                    carnivore_plot.set_clim(0, 1) # Ensure colormap is reset if no carnivores

                # Update population history
                x = range(len(self.herbivore_count_history))
                line1.set_data(x, self.herbivore_count_history)
                line2.set_data(x, self.carnivore_count_history)
                # Scale NDVI to be visible on same plot
                line3.set_data(x, [v * 100 for v in self.ndvi_mean_history])

                # Adjust y axis if needed
                max_pop = max(max(self.herbivore_count_history) if self.herbivore_count_history else 1,
                                max(self.carnivore_count_history) * 5 if self.carnivore_count_history else 1)
                ax4.set_ylim(0, max(max_pop * 1.1, 10))

                return ndvi_plot, herbivore_plot, carnivore_plot, line1, line2, line3

            anim = animation.FuncAnimation(fig, animate, frames=steps, init_func=init,
                                           interval=100, blit=False)

            if save_animation:
                writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save('ndvi_predator_prey_time.mp4', writer=writer)

            plt.tight_layout()
            plt.show()
        else:
            # Run without animation
            for i in range(steps):
                self.update(i)
                if i % 10 == 0:
                    print(f"Step {i}: {len(self.herbivores)} herbivores, {len(self.carnivores)} carnivores, NDVI Index: {self.current_ndvi_index}")

    def plot_results(self):
        """Plot the final simulation results in a web-based interface using Plotly"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Initial NDVI Distribution", "Final NDVI Distribution", 
                            "Final Animal Distribution (R: Carn., G: Herb.)", "Population and NDVI History"),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}], [{"type": "image"}, {"type": "xy"}]]
        )

        # Initial NDVI
        fig.add_trace(
            go.Heatmap(
                z=self.ndvi_grids[0],
                colorscale='YlGn',
                zmin=0,
                zmax=1,
                colorbar=dict(title="NDVI")
            ),
            row=1, col=1
        )

        # Final NDVI
        fig.add_trace(
            go.Heatmap(
                z=self.ndvi_grid,
                colorscale='YlGn',
                zmin=0,
                zmax=1,
                colorbar=dict(title="NDVI")
            ),
            row=1, col=2
        )

        # Final Animal Distribution
        combined_grid = np.zeros((self.grid_size, self.grid_size, 3))
        combined_grid[:, :, 1] = self.herbivore_grid / (
            np.max(self.herbivore_grid) if np.max(self.herbivore_grid) > 0 else 1)  # Green for herbivores
        combined_grid[:, :, 0] = self.carnivore_grid / (
            np.max(self.carnivore_grid) if np.max(self.carnivore_grid) > 0 else 1)  # Red for carnivores

        fig.add_trace(
            go.Image(z=(combined_grid * 255).astype(np.uint8)),
            row=2, col=1
        )

        # Population and NDVI History
        x = list(range(len(self.herbivore_count_history)))
        fig.add_trace(
            go.Scatter(x=x, y=self.herbivore_count_history, mode='lines', name='Herbivores', line=dict(color='green')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.carnivore_count_history, mode='lines', name='Carnivores', line=dict(color='red')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=x, y=[v * 100 for v in self.ndvi_mean_history], mode='lines', name='Mean NDVI x100', line=dict(color='blue', dash='dash')),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title="Simulation Results",
            height=800,
            width=1000,
            showlegend=True
        )

        # Display the plot
        fig.show()

    def get_plotly_graph_data(self):
        """Generate data for Plotly graphs."""
        return {
            'herbivore_count_history': self.herbivore_count_history,
            'carnivore_count_history': self.carnivore_count_history,
            'ndvi_mean_history': [v * 100 for v in self.ndvi_mean_history]  # Scale NDVI for better visualization
        }

# Example usage
if __name__ == "__main__":
    # Path to your folder containing the TIFF files
    tiff_folder = "data_tiff"

    # Create a dummy data_tiff folder and some dummy TIFF files if it doesn't exist
    if not os.path.exists(tiff_folder):
        os.makedirs(tiff_folder)
        dummy_ndvi_data = np.random.randint(-1000, 5000, size=(640, 672), dtype=np.int16)
        for i in range(19):
            tifffile.imwrite(os.path.join(tiff_folder, f"ndvi_{i+1}.tiff"), dummy_ndvi_data)
        print(f"Created dummy TIFF files in '{tiff_folder}'. Please replace with your actual data.")

    # Initialize simulation with TIFF data from the folder
    sim = NDVIPredatorPreySimulation(
        tiff_folder_path=tiff_folder,
        num_herbivores=1500,  # Increased to match larger grid
        num_carnivores=500,  # Increased to match larger grid
        ndvi_update_frequency=100,  # Update NDVI every 10 simulation steps
        rl_herbivores=True # Enable RL for herbivores
    )

    # Run simulation
    sim.run_simulation(steps=300, animate=True, save_animation=False)

    # Plot final results
    sim.plot_results()
