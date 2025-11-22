"""
Josh Zatz | UIC RAM Lab
Autonomous Driving Trust Calibration Game

The AI drives autonomously and avoids obstacles with randomized timing.
The user observes and presses LEFT/RIGHT when they would intervene to avoid a crash.
User input does NOT affect the game - it only calibrates trust in the background.
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from trust import TrustModel
import time
import sys

# ==================== GAME CONFIGURATION ====================
GAME_CONFIG = {
    # Display settings
    'screen_width': 1000,
    'screen_height': 800,
    'fps': 60,
    
    # Road and lane settings
    'num_lanes': 4,
    'lane_width': 80,
    'road_width': 320,
    'road_x_offset': 340,
    
    # AI car settings
    'car_width': 50,
    'car_height': 80,
    'car_start_lane': 1,
    'car_start_y': 450,
    
    # Movement settings
    'ai_speed': 4.0,
    'lane_change_speed': 5.0,
    
    # Traffic settings
    'traffic_spawn_rate': 0.01,
    'traffic_speed_min': 2.5,
    'traffic_speed_max': 5.0,
    'traffic_width': 50,
    'traffic_height': 80,
    
    # AI avoidance settings
    'min_avoidance_distance': 150,  # Minimum distance to start considering avoidance
    'max_avoidance_distance': 300,  # Maximum distance to start considering avoidance
    'danger_threshold': 200,  # Distance at which obstacle is considered dangerous
    
    # Trust Model Configuration
    'trust_prior': {'alpha': 2.0, 'beta': 2.0},
    'learning_rate': 1.5,
    'use_decay': False,
    'decay_rate': 0.0,
    'reward_params': {
        'taskSuccessWeight': 0.7,
        'trustWeight': 0.4,
        'trustTarget': 0.7,
        'interventionPenalty': -0.5,
        'decayPenalty': 0.0,
        'trustThreshold': 0.3
    },
    
    # Colors
    'colors': {
        'road': (50, 50, 50),
        'grass': (34, 139, 34),
        'lane_line': (255, 255, 0),
        'ai_car': (0, 150, 255),
        'traffic': (255, 50, 50),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'trust_bar': (0, 200, 0),
        'warning': (255, 200, 0),
        'bar_background': (200, 200, 200),
        'user_input_left': (255, 100, 100),
        'user_input_right': (100, 255, 100),
    }
}


# ==================== AI CAR (AUTONOMOUS) ====================
class AICar:
    """AI-controlled car that autonomously avoids obstacles"""
    
    def __init__(self, config):
        self.config = config
        self.current_lane = config['car_start_lane']
        self.target_lane = config['car_start_lane']
        self.y = config['car_start_y']
        self.speed = config['ai_speed']
        
        # Position tracking
        self.x = self._get_lane_center(self.current_lane)
        self.lane_change_progress = 0.0
        self.is_changing_lanes = False
        self.lane_change_start_x = self.x
        self.lane_change_target_x = self.x
        
        # Decision tracking
        self.last_decision_time = 0
        self.decision_cooldown = 0.5  # seconds between decisions
        
    def _get_lane_center(self, lane):
        return self.config['road_x_offset'] + (lane * self.config['lane_width']) + (self.config['lane_width'] // 2)
    
    def change_lane(self, target_lane):
        """Initiate a lane change"""
        if target_lane != self.current_lane and not self.is_changing_lanes:
            self.target_lane = target_lane
            self.is_changing_lanes = True
            self.lane_change_progress = 0.0
            self.lane_change_start_x = self.x
            self.lane_change_target_x = self._get_lane_center(target_lane)
    
    def update(self):
        """Update car position"""
        if self.is_changing_lanes:
            distance = abs(self.lane_change_target_x - self.lane_change_start_x)
            
            if distance < 0.1:
                self.is_changing_lanes = False
                self.x = self.lane_change_target_x
                self.current_lane = self.target_lane
                return
            
            self.lane_change_progress += self.config['lane_change_speed'] / distance
            
            if self.lane_change_progress >= 1.0:
                self.is_changing_lanes = False
                self.x = self.lane_change_target_x
                self.current_lane = self.target_lane
            else:
                # Smooth interpolation
                t = self.lane_change_progress
                smooth_t = t * t * (3 - 2 * t)  # Smoothstep
                self.x = self.lane_change_start_x + (self.lane_change_target_x - self.lane_change_start_x) * smooth_t
    
    def get_rect(self):
        """Get car bounds for collision detection (as dict instead of pygame.Rect)"""
        return {
            'x': int(self.x - self.config['car_width'] / 2),
            'y': int(self.y - self.config['car_height'] / 2),
            'width': self.config['car_width'],
            'height': self.config['car_height']
        }
    
    def make_avoidance_decision(self, obstacles, current_time):
        """
        Decide whether to avoid an obstacle, with randomized timing.
        Returns: (should_avoid, target_lane, obstacle_distance)
        """
        if current_time - self.last_decision_time < self.decision_cooldown:
            return False, None, None
        
        # Find closest obstacle in current lane
        closest_obstacle = None
        min_distance = float('inf')
        
        for obs in obstacles:
            if obs['lane'] == self.current_lane and obs['y'] < self.y:
                distance = self.y - obs['y']
                if distance < min_distance:
                    min_distance = distance
                    closest_obstacle = obs
        
        if closest_obstacle is None:
            return False, None, None
        
        # Check if obstacle is in avoidance range
        if min_distance < self.config['min_avoidance_distance']:
            return False, None, min_distance  # Too close, too late
        
        if min_distance > self.config['max_avoidance_distance']:
            return False, None, min_distance  # Too far, not worth avoiding yet
        
        # RANDOMIZED AVOIDANCE TIMING
        # Sample from a probability distribution based on distance
        # Closer = higher probability of avoiding
        distance_ratio = (self.config['max_avoidance_distance'] - min_distance) / \
                        (self.config['max_avoidance_distance'] - self.config['min_avoidance_distance'])
        
        # Probability increases as obstacle gets closer
        avoid_probability = distance_ratio ** 1.5  # Exponential increase
        
        if np.random.random() < avoid_probability:
            # Decide which lane to move to
            target_lane = self._find_safest_lane(obstacles)
            
            if target_lane != self.current_lane:
                self.last_decision_time = current_time
                return True, target_lane, min_distance
        
        return False, None, min_distance
    
    def _find_safest_lane(self, obstacles):
        """Find the safest lane to move to"""
        lane_safety = {}
        
        for lane in range(self.config['num_lanes']):
            # Count obstacles in each lane ahead
            obstacles_ahead = sum(1 for obs in obstacles 
                                 if obs['lane'] == lane and obs['y'] < self.y and 
                                 (self.y - obs['y']) < self.config['max_avoidance_distance'])
            
            # Prefer lanes with fewer obstacles
            lane_safety[lane] = -obstacles_ahead
            
            # Small penalty for distance from current lane
            lane_safety[lane] -= abs(lane - self.current_lane) * 0.1
        
        # Return safest lane
        return max(lane_safety, key=lane_safety.get)


# ==================== GAME ENVIRONMENT ====================
class GameEnvironment:
    """Manages traffic, rendering, and collision detection"""
    
    def __init__(self, config):
        self.config = config
        self.obstacles = []
        self.collision_count = 0
        self.obstacles_avoided = 0
        
        # Pygame setup
        self.screen = pygame.display.set_mode((config['screen_width'], config['screen_height']))
        pygame.display.set_caption('Autonomous Trust Calibration')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
    
    def spawn_traffic(self):
        """Randomly spawn traffic obstacles"""
        if np.random.random() < self.config['traffic_spawn_rate']:
            lane = np.random.randint(0, self.config['num_lanes'])
            x = self.config['road_x_offset'] + (lane * self.config['lane_width']) + (self.config['lane_width'] // 2)
            speed = np.random.uniform(self.config['traffic_speed_min'], self.config['traffic_speed_max'])
            
            self.obstacles.append({
                'x': x,
                'y': -self.config['traffic_height'],
                'lane': lane,
                'speed': speed
            })
    
    def update_traffic(self):
        """Update positions of all obstacles"""
        for obs in self.obstacles[:]:
            obs['y'] += obs['speed']
            
            # Remove if off screen
            if obs['y'] > self.config['screen_height'] + 100:
                self.obstacles.remove(obs)
    
    def check_collision(self, car_rect):
        """Check if car collided with any obstacle"""
        car_pygame_rect = pygame.Rect(car_rect['x'], car_rect['y'], car_rect['width'], car_rect['height'])
        
        for obs in self.obstacles:
            obs_rect = pygame.Rect(
                int(obs['x'] - self.config['traffic_width'] / 2),
                int(obs['y'] - self.config['traffic_height'] / 2),
                self.config['traffic_width'],
                self.config['traffic_height']
            )
            
            if car_pygame_rect.colliderect(obs_rect):
                return True, obs
        
        return False, None
    
    def remove_obstacle(self, obstacle):
        """Remove a specific obstacle"""
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
    
    def render(self, ai_car, trust_level, user_input_state, elapsed_time):
        """Render the game"""
        colors = self.config['colors']
        
        # Background
        self.screen.fill(colors['grass'])
        
        # Road
        road_rect = pygame.Rect(
            self.config['road_x_offset'],
            0,
            self.config['road_width'],
            self.config['screen_height']
        )
        pygame.draw.rect(self.screen, colors['road'], road_rect)
        
        # Lane lines
        for i in range(1, self.config['num_lanes']):
            x = self.config['road_x_offset'] + (i * self.config['lane_width'])
            for y in range(0, self.config['screen_height'], 40):
                pygame.draw.line(self.screen, colors['lane_line'], (x, y), (x, y + 20), 2)
        
        # Traffic
        for obs in self.obstacles:
            obs_rect = pygame.Rect(
                int(obs['x'] - self.config['traffic_width'] / 2),
                int(obs['y'] - self.config['traffic_height'] / 2),
                self.config['traffic_width'],
                self.config['traffic_height']
            )
            pygame.draw.rect(self.screen, colors['traffic'], obs_rect)
            pygame.draw.rect(self.screen, colors['white'], obs_rect, 2)
        
        # AI Car
        car_rect = ai_car.get_rect()
        car_pygame_rect = pygame.Rect(car_rect['x'], car_rect['y'], car_rect['width'], car_rect['height'])
        pygame.draw.rect(self.screen, colors['ai_car'], car_pygame_rect)
        pygame.draw.rect(self.screen, colors['white'], car_pygame_rect, 2)
        
        # User input indicator (if user pressed left or right recently)
        if user_input_state['active']:
            indicator_color = colors['user_input_left'] if user_input_state['direction'] == 'left' else colors['user_input_right']
            indicator_text = f'USER WOULD MOVE: {user_input_state["direction"].upper()}'
            
            # Draw indicator box
            indicator_rect = pygame.Rect(250, 100, 500, 50)
            pygame.draw.rect(self.screen, indicator_color, indicator_rect)
            pygame.draw.rect(self.screen, colors['white'], indicator_rect, 3)
            
            text = self.small_font.render(indicator_text, True, colors['white'])
            text_rect = text.get_rect(center=indicator_rect.center)
            self.screen.blit(text, text_rect)
        
        # Trust bar
        self._draw_trust_bar(trust_level)
        
        # Stats
        self._draw_stats(elapsed_time)
        
        pygame.display.flip()
    
    def _draw_trust_bar(self, trust_level):
        """Draw trust level bar"""
        colors = self.config['colors']
        bar_x, bar_y = 50, 50
        bar_width, bar_height = 300, 30
        
        # Background
        pygame.draw.rect(self.screen, colors['bar_background'], (bar_x, bar_y, bar_width, bar_height))
        
        # Trust fill
        fill_width = int(bar_width * trust_level)
        pygame.draw.rect(self.screen, colors['trust_bar'], (bar_x, bar_y, fill_width, bar_height))
        
        # Border
        pygame.draw.rect(self.screen, colors['white'], (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Text
        text = self.small_font.render(f'Trust: {trust_level:.2f}', True, colors['white'])
        self.screen.blit(text, (bar_x + 5, bar_y + 5))
    
    def _draw_stats(self, elapsed_time):
        """Draw game statistics"""
        colors = self.config['colors']
        stats_x, stats_y = 50, 700
        
        stats_text = [
            f'Time: {elapsed_time:.1f}s',
            f'Collisions: {self.collision_count}',
            f'Avoided: {self.obstacles_avoided}'
        ]
        
        for i, stat in enumerate(stats_text):
            text = self.small_font.render(stat, True, colors['white'])
            self.screen.blit(text, (stats_x, stats_y + i * 25))
    
    def tick(self, fps):
        """Control frame rate"""
        self.clock.tick(fps)


# ==================== TRUST MANAGER ====================
class TrustManager:
    """Manages trust model and tracks user predictions"""
    
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        
        # Initialize trust model
        self.trust_model = TrustModel(
            trustPrior=config['trust_prior'],
            observationModelParams={'observationType': 'binary'},
            transitionModelParams={
                'learningRate': config['learning_rate'],
                'isTimeVarying': config['use_decay'],
                'decayRate': config['decay_rate']
            },
            inferenceParams={},
            rewardParams=config['reward_params'],
            performanceTrackerParams={'windowSize': 100}
        )
        
        # History tracking
        self.history = {
            'time': [],
            'trust': [],
            'alpha': [],
            'beta': [],
            'user_interventions': [],
            'ai_actions': [],
            'outcomes': []
        }
        
        # User intervention tracking
        self.pending_interventions = []  # User interventions waiting for resolution
    
    def get_trust_level(self):
        """Get current trust level"""
        return self.trust_model.getTrust()
    
    def get_elapsed_time(self):
        """Get elapsed time since start"""
        return time.time() - self.start_time
    
    def record_user_intervention(self, direction, current_situation):
        """
        Record when user would intervene (press left/right to indicate they would move)
        direction: 'left' or 'right' - which direction user would move
        current_situation: dict with 'lane', 'obstacles_ahead', etc.
        """
        intervention = {
            'time': self.get_elapsed_time(),
            'direction': direction,
            'situation': current_situation,
            'resolved': False
        }
        self.pending_interventions.append(intervention)
        print(f"🖐️  User intervention: Would move {direction.upper()} (Trust: {self.get_trust_level():.3f})")
    
    def process_ai_avoidance(self, avoided_successfully, obstacle_distance):
        current_time = self.get_elapsed_time()
        
        # RESOLVE user interventions (if any)
        for intervention in self.pending_interventions:
            if not intervention['resolved'] and (current_time - intervention['time']) < 3.0:
                user_intervention_justified = not avoided_successfully
                
                context = {
                    'speed': 4.0,
                    'ttc': obstacle_distance / 50.0 if obstacle_distance else 1.0
                }
                
                trust_result = self.trust_model.updateTrust(
                    robotAction='avoid_obstacle',
                    robotPerformance=avoided_successfully,
                    humanResponse=True,  # User signaled intervention
                    context=context
                )
                
                intervention['resolved'] = True
                intervention['justified'] = user_intervention_justified
                intervention['ai_succeeded'] = avoided_successfully
                
                self.history['user_interventions'].append({
                    'time': current_time,
                    'justified': user_intervention_justified,
                    'ai_succeeded': avoided_successfully
                })
        
        # ✅ ADD THIS: Always update trust for AI actions, even without user input
        if not any(i for i in self.pending_interventions 
                if not i['resolved'] and (current_time - i['time']) < 3.0):
            # No recent user intervention, update trust for autonomous action
            context = {
                'speed': 4.0,
                'ttc': obstacle_distance / 50.0 if obstacle_distance else 1.0
            }
            
            trust_result = self.trust_model.updateTrust(
                robotAction='avoid_obstacle',
                robotPerformance=avoided_successfully,
                humanResponse=False,  # No user intervention = implicit trust
                context=context
            )
        
        # Clean up old interventions
        self.pending_interventions = [i for i in self.pending_interventions 
                                    if not i['resolved'] or (current_time - i['time']) < 5.0]
        
        # Record history
        self._record_history()
    
    def process_collision(self):
        """Process collision event"""
        context = {'speed': 4.0, 'ttc': 0.0}
        
        trust_result = self.trust_model.updateTrust(
            robotAction='collision',
            robotPerformance=False,
            humanResponse=False,
            context=context
        )
        
        self._record_history()
        
        print(f"💥 COLLISION! Trust: {self.get_trust_level():.3f}")
        
        return trust_result
    
    def _record_history(self):
        """Record current state to history"""
        trust_dist = self.trust_model.getTrustDistribution()
        
        self.history['time'].append(self.get_elapsed_time())
        self.history['trust'].append(trust_dist['mean'])
        self.history['alpha'].append(trust_dist['alpha'])
        self.history['beta'].append(trust_dist['beta'])
    
    def show_plots(self):
        """Show trust evolution plots"""
        if len(self.history['time']) < 2:
            print("Not enough data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Trust over time
        ax = axes[0, 0]
        ax.plot(self.history['time'], self.history['trust'], 'b-', linewidth=2, label='Trust')
        ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target')
        ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Trust Level')
        ax.set_title('Trust Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Alpha/Beta
        ax = axes[0, 1]
        ax.plot(self.history['time'], self.history['alpha'], 'b-', label='Alpha', linewidth=2)
        ax.plot(self.history['time'], self.history['beta'], 'r-', label='Beta', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Beta Distribution Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # User interventions
        ax = axes[1, 0]
        if self.history['user_interventions']:
            times = [i['time'] for i in self.history['user_interventions']]
            justified = [1 if i['justified'] else 0 for i in self.history['user_interventions']]
            ax.scatter(times, justified, c=['g' if j else 'r' for j in justified], s=50, alpha=0.6)
            ax.set_ylim([-0.1, 1.1])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Unnecessary', 'Justified'])
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('User Intervention')
        ax.set_title('User Intervention Assessment')
        ax.grid(True, alpha=0.3)
        
        # Summary stats
        ax = axes[1, 1]
        ax.axis('off')
        
        if self.history['user_interventions']:
            total_interventions = len(self.history['user_interventions'])
            justified_interventions = sum(1 for i in self.history['user_interventions'] if i['justified'])
            justification_rate = justified_interventions / total_interventions if total_interventions > 0 else 0
            
            summary_text = f"""
            SUMMARY STATISTICS
            
            Total User Interventions: {total_interventions}
            Justified Interventions: {justified_interventions}
            Justification Rate: {justification_rate:.1%}
            
            (Justified = AI crashed, user was right to doubt)
            (Unnecessary = AI avoided, user doubted needlessly)
            
            Final Trust: {self.history['trust'][-1]:.3f}
            Final Alpha: {self.history['alpha'][-1]:.2f}
            Final Beta: {self.history['beta'][-1]:.2f}
            """
        else:
            summary_text = "No user interventions recorded"
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.show()


# ==================== MAIN GAME LOOP ====================
def run_game(config):
    ai_car = AICar(config)
    environment = GameEnvironment(config)
    trust_manager = TrustManager(config)
    
    # User input state
    user_input_state = {
        'active': False,
        'direction': None,
        'time': 0
    }
    
    running = True
    print("Game started! Press LEFT/RIGHT when you would intervene. ESC to quit.\n")
    
    while running:
        current_time = trust_manager.get_elapsed_time()
        
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    # User would move left to avoid crash
                    situation = {
                        'lane': ai_car.current_lane,
                        'time': current_time
                    }
                    trust_manager.record_user_intervention('left', situation)
                    user_input_state = {'active': True, 'direction': 'left', 'time': current_time}
                elif event.key == pygame.K_RIGHT:
                    # User would move right to avoid crash
                    situation = {
                        'lane': ai_car.current_lane,
                        'time': current_time
                    }
                    trust_manager.record_user_intervention('right', situation)
                    user_input_state = {'active': True, 'direction': 'right', 'time': current_time}
        
        # Clear user input indicator after 1 second
        if user_input_state['active'] and (current_time - user_input_state['time']) > 1.0:
            user_input_state = {'active': False, 'direction': None, 'time': 0}
        
        # Spawn and update traffic
        environment.spawn_traffic()
        environment.update_traffic()
        
        # AI decision making
        should_avoid, target_lane, distance = ai_car.make_avoidance_decision(
            environment.obstacles, current_time
        )
        
        if should_avoid and target_lane is not None:
            ai_car.change_lane(target_lane)
            print(f"🤖 AI avoiding: Lane {ai_car.current_lane} → {target_lane} (distance: {distance:.1f})")
            environment.obstacles_avoided += 1
            trust_manager.process_ai_avoidance(avoided_successfully=True, obstacle_distance=distance)
        
        # Update AI car
        ai_car.update()
        
        # Check collision
        collision, collided_obstacle = environment.check_collision(ai_car.get_rect())
        if collision:
            environment.collision_count += 1
            trust_manager.process_collision()
            trust_manager.process_ai_avoidance(avoided_successfully=False, obstacle_distance=0)
            environment.remove_obstacle(collided_obstacle)
        
        # Render
        environment.render(
            ai_car,
            trust_manager.get_trust_level(),
            user_input_state,
            current_time
        )
        
        environment.tick(config['fps'])
    
    # Cleanup
    pygame.quit()
    print("\n" + "="*70)
    print("GAME OVER")
    print("="*70)
    print(f"Collisions: {environment.collision_count}")
    print(f"Obstacles Avoided: {environment.obstacles_avoided}")
    print(f"Final Trust: {trust_manager.get_trust_level():.3f}")
    print("="*70)
    
    trust_manager.show_plots()


# ==================== MAIN ====================
if __name__ == "__main__":
    print("="*70)
    print("AUTONOMOUS DRIVING TRUST CALIBRATION")
    print("="*70)
    print("\nControls:")
    print("  LEFT ARROW:  Press when you would move left to avoid crash")
    print("  RIGHT ARROW: Press when you would move right to avoid crash")
    print("  ESC:         Quit game")
    print("\nObjective:")
    print("  The AI drives autonomously. You OBSERVE only.")
    print("  Press LEFT/RIGHT when you would intervene to avoid a crash.")
    print("  Your inputs do NOT affect the game - they calibrate trust.")
    print("  The system learns whether you trust the AI's decisions.")
    print("="*70 + "\n")
    
    try:
        pygame.init()
        run_game(GAME_CONFIG)
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        sys.exit(0)
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR:")
        print("="*70)
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*70)
        sys.exit(1)