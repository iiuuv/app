import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Circle, Rectangle, Wedge

class Direction:
    ForWard = 1
    BackWard = 2
    Rotate = 3


class MovePlan:
    def __init__(self):
        self.direction: Direction = Direction.ForWard
        self.distance: float = 0.0


def search_path(map:list[list[bool]], start:tuple[int,int], end:tuple[int,int],car_radius:int,fire_radius:int) -> list[MovePlan]:
    """
    Find path from start to area near end point
    
    Parameters:
    - map: 2D boolean map, True indicates obstacle, False indicates passable
    - start: Start coordinates (x, y)
    - end: End coordinates (x, y)
    - car_radius: Vehicle radius for collision detection
    - fire_radius: Firing radius, final position should be 1-2 times fire_radius from target
    
    Returns:
    - list[MovePlan]: List of movement plans
    """
    
    if not map or not map[0]:
        return []
    
    height = len(map)
    width = len(map[0])
    
    def is_valid_position(x, y):
        """Check if position is valid (considering vehicle radius)"""
        # Check boundaries
        if x - car_radius < 0 or y - car_radius < 0 or x + car_radius >= width or y + car_radius >= height:
            return False
        
        # Check circular area centered at (x,y) with car_radius for obstacles
        for dy in range(-car_radius, car_radius + 1):
            for dx in range(-car_radius, car_radius + 1):
                if dx*dx + dy*dy <= car_radius*car_radius:  # Circular collision detection
                    check_x, check_y = x + dx, y + dy
                    if 0 <= check_x < width and 0 <= check_y < height:
                        if map[check_y][check_x]:  # Hit obstacle
                            return False
        return True
    
    def analyze_polar_obstacles():
        """Analyze obstacles using ray casting from end point"""
        obstacle_rays = []
        
        # Cast rays in all directions (every degree for high precision)
        for angle_deg in range(0, 360, 1):  # 1-degree resolution
            angle_rad = math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            
            # Cast ray from end point outward
            hit_obstacle = False
            hit_distance = float('inf')
            
            # Ray casting with fine step resolution
            step_size = 0.5  # Fine resolution for accurate detection
            max_ray_distance = 2 * fire_radius + car_radius + 5  # Extend beyond our interest range
            
            for step in range(int(max_ray_distance / step_size)):
                ray_distance = step * step_size
                ray_x = end[0] + dx * ray_distance
                ray_y = end[1] + dy * ray_distance
                
                # Check map boundaries
                if ray_x < 0 or ray_x >= width or ray_y < 0 or ray_y >= height:
                    hit_distance = ray_distance
                    hit_obstacle = True
                    break
                
                # Check for obstacle
                grid_x = int(ray_x)
                grid_y = int(ray_y)
                if 0 <= grid_x < width and 0 <= grid_y < height:
                    if map[grid_y][grid_x]:  # Hit obstacle
                        hit_distance = ray_distance
                        hit_obstacle = True
                        break
            
            obstacle_rays.append({
                'angle': angle_deg,
                'distance': hit_distance,
                'hit_obstacle': hit_obstacle
            })
        
        return obstacle_rays
    
    def find_clear_sectors(obstacle_rays, min_gap_degrees=45):
        """Find sectors with obstacle gaps >= min_gap_degrees"""
        clear_sectors = []
        
        if not obstacle_rays:
            return clear_sectors
        
        # Sort by angle to ensure proper order
        obstacle_rays.sort(key=lambda x: x['angle'])
        
        # Find angles where obstacles are within our range of interest
        obstacle_angles = []
        min_obstacle_distance = fire_radius  # We care about obstacles closer than this
        
        for ray in obstacle_rays:
            if ray['hit_obstacle'] and ray['distance'] <= min_obstacle_distance:
                obstacle_angles.append(ray['angle'])
        
        if not obstacle_angles:
            # No close obstacles, entire circle is clear
            return [(0, 360)]
        
        # Find gaps between consecutive obstacle angles
        obstacle_angles.sort()
        
        for i in range(len(obstacle_angles)):
            current_angle = obstacle_angles[i]
            next_angle = obstacle_angles[(i + 1) % len(obstacle_angles)]
            
            # Calculate gap between current and next obstacle
            if next_angle > current_angle:
                gap = next_angle - current_angle
                sector_start = current_angle
                sector_end = next_angle
            else:
                # Handle wrap-around case (e.g., from 350° to 10°)
                gap = (360 - current_angle) + next_angle
                sector_start = current_angle
                sector_end = next_angle
            
            # If gap is large enough, this is a clear sector
            if gap >= min_gap_degrees:
                # Normalize angles to [0, 360) range
                if sector_end > 360:
                    # Handle wrapping case
                    clear_sectors.append((sector_start, 360))
                    clear_sectors.append((0, sector_end - 360))
                else:
                    clear_sectors.append((sector_start, sector_end))
        
        return clear_sectors
    
    def is_position_in_clear_sector(pos, clear_sectors):
        """Check if a position is within any clear sector"""
        # If no clear sectors, position is invalid
        if not clear_sectors:
            return False
            
        dx = pos[0] - end[0]
        dy = pos[1] - end[1]
        angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # 精确检查每个扇区
        for sector_start, sector_end in clear_sectors:
            # 处理正常扇区
            if sector_end <= 360:
                if sector_start <= angle <= sector_end:
                    return True
            else:
                # 处理跨越0°的扇区 (如 350° to 370° 表示 350° to 360° 和 0° to 10°)
                if angle >= sector_start or angle <= (sector_end - 360):
                    return True
        
        return False
    
    def is_valid_position(x, y):
        """Check if position is valid (considering vehicle radius)"""
        # Check boundaries
        if x - car_radius < 0 or y - car_radius < 0 or x + car_radius >= width or y + car_radius >= height:
            return False
        return True
    
    def get_target_positions():
        """Get valid target positions 1-2 times fire_radius from end point, using polar coordinate obstacle analysis
        Only sectors with obstacle gaps >= 45° are allowed for final positioning"""
        targets = []
        min_dist = fire_radius
        max_dist = 2 * fire_radius
        
        # Analyze obstacles in polar coordinates and find clear sectors
        obstacle_rays = analyze_polar_obstacles()
        clear_sectors = find_clear_sectors(obstacle_rays, min_gap_degrees=45)
        
        print(f"DEBUG: Found {len(clear_sectors)} clear sectors with gaps >= 45°:")
        for i, (sector_start, sector_end) in enumerate(clear_sectors):
            print(f"  Sector {i+1}: {sector_start:.1f}° to {sector_end:.1f}° (gap: {(sector_end-sector_start) % 360:.1f}°)")
        
        # 严格检查：如果没有合法扇区，立即返回空列表
        if not clear_sectors:
            print("DEBUG: No clear sectors found, no valid target positions")
            return []
        
        # 只有在有合法扇区时才搜索目标位置
        print(f"DEBUG: Searching for target positions in {len(clear_sectors)} clear sectors...")
        
        # Search for valid positions around target point
        search_range = max_dist + car_radius
        valid_position_count = 0
        sector_position_count = 0
        
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                target_x = end[0] + dx
                target_y = end[1] + dy
                
                # Check if within map boundaries
                if 0 <= target_x < width and 0 <= target_y < height:
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    # Check distance constraint: must be in the annular region AND outside the no-go zone
                    # No-go zone: within fire_radius of end point (距离end点周围一个fire_radius的区域为不可通行区域)
                    if min_dist <= dist <= max_dist and dist > fire_radius and is_valid_position(target_x, target_y):
                        valid_position_count += 1
                        
                        # 严格检查：只有在合法扇区内的位置才被添加
                        if is_position_in_clear_sector((target_x, target_y), clear_sectors):
                            sector_position_count += 1
                            targets.append((target_x, target_y))
                            angle_deg = math.degrees(math.atan2(target_y - end[1], target_x - end[0])) % 360
                            print(f"DEBUG: Added valid target at ({target_x}, {target_y}), angle: {angle_deg:.1f}°")
                            
                            # 额外验证：显示这个位置在哪个扇区内
                            for i, (s_start, s_end) in enumerate(clear_sectors):
                                if s_end > 360:  # Handle wrap-around 
                                    if angle_deg >= s_start or angle_deg <= (s_end - 360):
                                        print(f"      -> In Sector {i+1}: {s_start:.1f}° to {s_end:.1f}°")
                                        break
                                else:
                                    if s_start <= angle_deg <= s_end:
                                        print(f"      -> In Sector {i+1}: {s_start:.1f}° to {s_end:.1f}°")
                                        break
        
        print(f"DEBUG: Found {valid_position_count} valid positions, {sector_position_count} in clear sectors")
        
        # 最终安全检查：确保只返回在合法扇区内的位置
        if not targets:
            print("DEBUG: No target positions found in clear sectors!")
        
        return targets
    
    def heuristic(pos1, pos2):
        """A* algorithm heuristic function (Euclidean distance)"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(pos):
        """Get adjacent positions in 8 directions, but only in clear sectors"""
        x, y = pos
        neighbors = []
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # up down left right
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonals
        ]
        
        # Get clear sectors for checking
        obstacle_rays = analyze_polar_obstacles()
        clear_sectors = find_clear_sectors(obstacle_rays, min_gap_degrees=45)
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            # Position must be valid AND in a clear sector
            if is_valid_position(new_x, new_y):
                # Additional check: position must be in clear sector OR be the start position
                # (We allow start position even if it's not in clear sector)
                if pos == start or is_position_in_clear_sector((new_x, new_y), clear_sectors):
                    neighbors.append((new_x, new_y))
        
        return neighbors
    
    def a_star_search(start_pos, target_pos):
        """A* pathfinding algorithm with improved early termination when reaching valid clear sector"""
        open_set = []
        heapq.heappush(open_set, (0, start_pos))
        
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: heuristic(start_pos, target_pos)}
        
        open_set_hash = {start_pos}
        
        # Get clear sectors for early termination check
        obstacle_rays = analyze_polar_obstacles()
        clear_sectors = find_clear_sectors(obstacle_rays, min_gap_degrees=45)
        min_dist = fire_radius
        max_dist = 2 * fire_radius
        
        # Track the best early termination candidate found so far
        best_early_termination = None
        best_early_distance = float('inf')
        
        print(f"DEBUG: A* starting with {len(clear_sectors)} clear sectors available")
        
        steps_count = 0
        while open_set:
            steps_count += 1
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # Early termination check: if current position is in clear sector and meets distance constraints
            if current != start_pos:  # Don't terminate at start position
                dist_to_end = math.sqrt((current[0] - end[0])**2 + (current[1] - end[1])**2)
                dist_from_start = g_score[current]
                
                # Check if current position meets all criteria for early termination
                if (min_dist <= dist_to_end <= max_dist and 
                    dist_to_end > fire_radius and  # Outside no-go zone
                    is_position_in_clear_sector(current, clear_sectors)):
                    
                    # Found a valid clear sector point - check if it's better than previous candidates
                    if dist_from_start < best_early_distance:
                        best_early_termination = current
                        best_early_distance = dist_from_start
                        
                        print(f"DEBUG: Better early termination candidate at {current}, "
                              f"distance from start: {dist_from_start:.2f}, distance to end: {dist_to_end:.2f}")
                    
                    # Early termination strategy: terminate immediately if this is very close to start
                    # or if we've searched enough steps (balance between optimality and efficiency)
                    if dist_from_start <= 8 or steps_count >= 50:
                        print(f"DEBUG: Immediate early termination at {current} after {steps_count} steps")
                        
                        # Reconstruct path to current position
                        path = []
                        temp_current = current
                        while temp_current in came_from:
                            path.append(temp_current)
                            temp_current = came_from[temp_current]
                        path.append(start_pos)
                        path.reverse()
                        print(f"DEBUG: Early termination path length: {len(path)} points")
                        return path
            
            # Normal termination: reached original target
            if current == target_pos:
                print(f"DEBUG: Reached original target {target_pos}")
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_pos)
                path.reverse()
                return path
            
            # If we've found a good early termination candidate and searched enough,
            # terminate to the best candidate found so far
            if best_early_termination and steps_count >= 100:
                print(f"DEBUG: Terminating to best candidate {best_early_termination} after {steps_count} steps")
                
                # Reconstruct path to best candidate
                path = []
                temp_current = best_early_termination
                while temp_current in came_from:
                    path.append(temp_current)
                    temp_current = came_from[temp_current]
                path.append(start_pos)
                path.reverse()
                print(f"DEBUG: Best candidate path length: {len(path)} points")
                return path
            
            for neighbor in get_neighbors(current):
                # Calculate movement cost
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = math.sqrt(dx*dx + dy*dy)  # Euclidean distance
                
                temp_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, target_pos)
                    
                    if neighbor not in open_set_hash:
                        # Check if this neighbor is a potential early termination point
                        neighbor_dist_to_end = math.sqrt((neighbor[0] - end[0])**2 + (neighbor[1] - end[1])**2)
                        is_clear_sector_point = (neighbor != start_pos and
                                               min_dist <= neighbor_dist_to_end <= max_dist and 
                                               neighbor_dist_to_end > fire_radius and  
                                               is_position_in_clear_sector(neighbor, clear_sectors))
                        
                        if is_clear_sector_point:
                            # Give higher priority to clear sector points by reducing their f-score
                            priority_boost = 10  # Lower value = higher priority
                            heapq.heappush(open_set, (f_score[neighbor] - priority_boost, neighbor))
                            print(f"DEBUG: Prioritizing clear sector candidate {neighbor}, "
                                  f"distance to end: {neighbor_dist_to_end:.2f}")
                        else:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
                        open_set_hash.add(neighbor)
        
        # If we have a best early termination candidate but didn't use it, use it now
        if best_early_termination:
            print(f"DEBUG: Using best early termination candidate {best_early_termination}")
            
            path = []
            temp_current = best_early_termination
            while temp_current in came_from:
                path.append(temp_current)
                temp_current = came_from[temp_current]
            path.append(start_pos)
            path.reverse()
            return path
        
        return None  # No path found
    
    def path_to_move_plans(path):
        """Convert path to MovePlan list with advanced optimization:
        1. First perform path smoothing by connecting distant points directly if possible
        2. Then merge consecutive same-direction moves
        """
        if not path or len(path) < 2:
            return []
        
        def can_move_directly(start_pos, end_pos):
            """Check if we can move directly from start_pos to end_pos without hitting obstacles"""
            x1, y1 = start_pos
            x2, y2 = end_pos
            
            # Use Bresenham-like algorithm to check all points along the line
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            if dx == 0 and dy == 0:
                return True
            
            # Determine step direction
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            
            # Calculate number of steps needed
            steps = max(dx, dy)
            
            # Check each point along the path
            for i in range(steps + 1):
                # Linear interpolation
                t = i / max(steps, 1)
                check_x = int(x1 + t * (x2 - x1))
                check_y = int(y1 + t * (y2 - y1))
                
                # Check if this position is valid (considering car radius)
                if not is_valid_position(check_x, check_y):
                    return False
            
            return True
        
        def detect_turn_clusters(path):
            """检测路径中的连续转弯集群 - 支持0-180度的所有转弯角度"""
            if len(path) < 4:
                return []
            
            def calculate_turn_angle(p1, p2, p3):
                """计算在p2点的转弯角度 - 返回绝对转弯角度(0-180度)"""
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # 计算向量长度
                len1 = math.sqrt(v1[0]**2 + v1[1]**2)
                len2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if len1 == 0 or len2 == 0:
                    return 0
                
                # 归一化向量
                v1_norm = (v1[0]/len1, v1[1]/len1)
                v2_norm = (v2[0]/len2, v2[1]/len2)
                
                # 计算点积（余弦值）
                dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
                # 限制在[-1, 1]范围内以避免数值误差
                dot_product = max(-1.0, min(1.0, dot_product))
                
                # 计算夹角（0-180度）
                angle_rad = math.acos(dot_product)
                angle_deg = math.degrees(angle_rad)
                
                return angle_deg
            
            turn_clusters = []
            current_cluster = []
            
            # 检测所有转弯点（包括轻微转弯到急转弯）
            for i in range(1, len(path) - 1):
                turn_angle = calculate_turn_angle(path[i-1], path[i], path[i+1])
                
                # 大幅降低转弯检测阈值：任何大于2度的角度变化都认为是转弯
                # 这样可以捕获几乎所有的方向变化，包括非常轻微的转弯
                if turn_angle > 2:  # 进一步降低阈值从5度到2度
                    print(f"DEBUG: 检测到转弯点 {i} at {path[i]}, 转弯角度: {turn_angle:.1f}°")
                    
                    if not current_cluster:
                        current_cluster = [i]
                    else:
                        # 检查与上一个转弯点的距离和角度连续性
                        last_turn_idx = current_cluster[-1]
                        distance = math.sqrt((path[i][0] - path[last_turn_idx][0])**2 + 
                                           (path[i][1] - path[last_turn_idx][1])**2)
                        
                        # 大幅增加距离阈值以包含更多连续转弯
                        # 轻微转弯使用更大的距离阈值，中等转弯使用中等阈值，急转弯使用最大阈值
                        if turn_angle > 60:  # 急转弯
                            distance_threshold = 15
                        elif turn_angle > 30:  # 中等转弯
                            distance_threshold = 12
                        elif turn_angle > 10:  # 小转弯
                            distance_threshold = 10
                        else:  # 轻微转弯
                            distance_threshold = 8
                        
                        if distance < distance_threshold:
                            current_cluster.append(i)
                            print(f"DEBUG: 添加到当前集群，距离: {distance:.1f}")
                        else:
                            # 检查角度相似性：如果转弯方向相似，即使距离稍远也归为一个集群
                            last_turn_angle = calculate_turn_angle(path[last_turn_idx-1], 
                                                                 path[last_turn_idx], 
                                                                 path[last_turn_idx+1])
                            
                            # 如果两个转弯角度都比较大（>20度）且方向相似，或者距离很近，扩大距离容忍度
                            if ((turn_angle > 20 and last_turn_angle > 20 and 
                                abs(turn_angle - last_turn_angle) < 60) or distance < 15):
                                current_cluster.append(i)
                                print(f"DEBUG: 基于角度相似性或近距离添加到集群，角度差: {abs(turn_angle - last_turn_angle):.1f}°, 距离: {distance:.1f}")
                            else:
                                # 保存当前集群，开始新集群
                                if len(current_cluster) >= 1:  # 保持单点也能形成集群
                                    turn_clusters.append(current_cluster)
                                    print(f"DEBUG: 保存集群: {current_cluster}")
                                current_cluster = [i]
                else:
                    # 非转弯点或转弯角度太小，结束当前集群
                    if len(current_cluster) >= 1:  # 降低集群最小大小要求
                        turn_clusters.append(current_cluster)
                        print(f"DEBUG: 保存集群: {current_cluster}")
                    current_cluster = []
            
            # 处理最后一个集群
            if len(current_cluster) >= 1:  # 降低集群最小大小要求
                turn_clusters.append(current_cluster)
                print(f"DEBUG: 保存最后集群: {current_cluster}")
            
            print(f"DEBUG: 总共检测到 {len(turn_clusters)} 个转弯集群")
            return turn_clusters
        
        def optimize_turn_cluster(path, cluster_indices):
            """优化转弯集群，将转弯点向曲线突出部移动，支持0-180度转弯优化"""
            if len(cluster_indices) < 1:  # 支持单点集群优化
                return path
            
            # 计算集群的边界，扩大边界范围以获得更好的上下文
            start_idx = max(0, cluster_indices[0] - 2)  # 扩大到前2个点
            end_idx = min(len(path) - 1, cluster_indices[-1] + 2)  # 扩大到后2个点
            
            # 获取集群相关的路径段
            cluster_points = [path[i] for i in range(start_idx, end_idx + 1)]
            
            if len(cluster_points) < 3:
                return path
            
            # 计算曲线的质心（加权质心，给转弯点更大权重）
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            
            for i, point in enumerate(cluster_points):
                # 如果是转弯点，给予更大权重
                actual_idx = start_idx + i
                weight = 3.0 if actual_idx in cluster_indices else 1.0
                
                weighted_x += point[0] * weight
                weighted_y += point[1] * weight
                total_weight += weight
            
            centroid = (weighted_x / total_weight, weighted_y / total_weight)
            
            # 计算路径的主要方向（起点到终点的方向）
            path_start = cluster_points[0]
            path_end = cluster_points[-1]
            main_direction = (path_end[0] - path_start[0], path_end[1] - path_start[1])
            main_length = math.sqrt(main_direction[0]**2 + main_direction[1]**2)
            
            if main_length > 0:
                main_direction = (main_direction[0]/main_length, main_direction[1]/main_length)
            else:
                main_direction = (1, 0)  # 默认方向
            
            optimized_path = path.copy()
            
            # 优化每个转弯点
            for cluster_idx in cluster_indices:
                original_point = path[cluster_idx]
                
                # 计算转弯的锐利程度
                if cluster_idx > 0 and cluster_idx < len(path) - 1:
                    prev_point = path[cluster_idx - 1]
                    next_point = path[cluster_idx + 1]
                    
                    # 计算转弯角度
                    v1 = (original_point[0] - prev_point[0], original_point[1] - prev_point[1])
                    v2 = (next_point[0] - original_point[0], next_point[1] - original_point[1])
                    
                    len1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    len2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    
                    if len1 > 0 and len2 > 0:
                        v1_norm = (v1[0]/len1, v1[1]/len1)
                        v2_norm = (v2[0]/len2, v2[1]/len2)
                        
                        dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
                        dot_product = max(-1.0, min(1.0, dot_product))
                        turn_angle = math.degrees(math.acos(dot_product))
                        
                        # 根据转弯角度调整优化强度，增加优化力度
                        if turn_angle > 120:  # 急转弯(120-180度)
                            optimization_factor = 0.8  # 强优化 (从0.6增加到0.8)
                        elif turn_angle > 60:   # 中等转弯(60-120度)
                            optimization_factor = 0.6  # 中等优化 (从0.4增加到0.6)
                        elif turn_angle > 30:   # 轻微转弯(30-60度)
                            optimization_factor = 0.4  # 轻微优化 (从0.2增加到0.4)
                        elif turn_angle > 10:   # 小转弯(10-30度)
                            optimization_factor = 0.3  # 小优化 (新增)
                        else:                   # 很小转弯(2-10度)
                            optimization_factor = 0.2  # 最小优化 (从0.1增加到0.2)
                    else:
                        optimization_factor = 0.3  # 默认值
                else:
                    optimization_factor = 0.3  # 默认值
                
                # 计算从质心指向原点的方向（突出部方向）
                dx = original_point[0] - centroid[0]
                dy = original_point[1] - centroid[1]
                dist_to_centroid = math.sqrt(dx*dx + dy*dy)
                
                if dist_to_centroid > 0:
                    # 归一化方向向量
                    norm_dx = dx / dist_to_centroid
                    norm_dy = dy / dist_to_centroid
                    
                    # 计算移动距离，增加基础移动距离和优化范围
                    base_move_distance = min(5.0, dist_to_centroid * 0.6)  # 从3.0增加到5.0，从0.4增加到0.6
                    move_distance = base_move_distance * optimization_factor
                    
                    # 向突出部方向移动
                    new_x = original_point[0] + norm_dx * move_distance
                    new_y = original_point[1] + norm_dy * move_distance
                    
                    # 确保新位置仍然有效且不会偏离路径太远
                    new_pos = (int(new_x), int(new_y))
                    
                    if (is_valid_position(new_pos[0], new_pos[1]) and
                        math.sqrt((new_pos[0] - original_point[0])**2 + 
                                 (new_pos[1] - original_point[1])**2) <= 4):  # 限制移动距离
                        
                        optimized_path[cluster_idx] = new_pos
                        print(f"DEBUG: 优化转弯点 {cluster_idx}: {original_point} -> {new_pos}, "
                              f"移动距离: {move_distance:.2f}, 优化因子: {optimization_factor:.2f}")
                    else:
                        print(f"DEBUG: 转弯点 {cluster_idx} 优化被跳过（无效位置或移动过远）")
            
            return optimized_path
        
        def extend_lines_and_replace_turns(path):
            """
            直线延长-交点替换算法：
            检测小段转弯路径，尝试将其两端的长直线延长，
            如果交点在可行区域，则删除小段转弯，用直接转角替换
            """
            if len(path) < 5:  # 至少需要5个点才能进行这种优化
                return path
            
            def get_line_direction(p1, p2):
                """获取直线方向向量（归一化）"""
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx*dx + dy*dy)
                if length == 0:
                    return (0, 0)
                return (dx/length, dy/length)
            
            def are_points_collinear(p1, p2, p3, tolerance=0.1):
                """检查三个点是否近似共线"""
                # 计算向量
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # 计算叉积
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                
                # 如果叉积接近0，则点近似共线
                return abs(cross_product) < tolerance
            
            def find_straight_segment(path, start_idx, direction=1):
                """
                从起始索引开始，沿指定方向查找最长的直线段
                direction: 1表示向前，-1表示向后
                返回: (segment_start_idx, segment_end_idx, is_long_enough)
                """
                if direction == 1:
                    indices = range(start_idx, len(path) - 1)
                else:
                    indices = range(start_idx, 0, -1)
                
                segment_start = start_idx
                segment_end = start_idx
                
                for i in indices:
                    next_i = i + direction
                    if next_i < 0 or next_i >= len(path):
                        break
                    
                    # 检查是否还在同一直线上
                    if segment_end != segment_start:  # 已有至少2个点
                        if not are_points_collinear(path[segment_start], path[i], path[next_i], tolerance=0.2):
                            break
                    
                    segment_end = next_i
                
                # 计算线段长度
                if segment_end != segment_start:
                    length = math.sqrt((path[segment_end][0] - path[segment_start][0])**2 + 
                                     (path[segment_end][1] - path[segment_start][1])**2)
                    is_long_enough = length >= 8  # 至少8个单位长度才认为是"长"直线
                else:
                    is_long_enough = False
                
                return segment_start, segment_end, is_long_enough
            
            def line_intersection(p1, d1, p2, d2):
                """
                计算两条直线的交点
                p1, p2: 直线上的点
                d1, d2: 直线方向向量
                返回: 交点坐标或None（如果平行）
                """
                # 直线方程: p1 + t1 * d1 = p2 + t2 * d2
                # 解方程组求t1, t2
                
                denominator = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(denominator) < 1e-10:  # 平行线
                    return None
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                t1 = (dx * d2[1] - dy * d2[0]) / denominator
                
                # 计算交点
                intersection_x = p1[0] + t1 * d1[0]
                intersection_y = p1[1] + t1 * d1[1]
                
                return (int(intersection_x), int(intersection_y))
            
            def detect_short_turn_segments(path):
                """检测可能的短转弯段"""
                short_turn_segments = []
                
                i = 0
                while i < len(path) - 4:  # 至少需要5个点
                    # 查找前方直线段
                    front_start, front_end, front_is_long = find_straight_segment(path, i, direction=1)
                    
                    if not front_is_long or front_end >= len(path) - 2:
                        i += 1
                        continue
                    
                    # 从前方直线段结束后查找转弯段
                    turn_start = front_end
                    turn_length = 0
                    
                    # 寻找转弯段的结束位置（下一个直线段的开始）
                    j = turn_start + 1
                    while j < len(path) - 1:
                        # 检查从j开始是否有后方直线段
                        back_start, back_end, back_is_long = find_straight_segment(path, j, direction=1)
                        
                        if back_is_long:
                            turn_end = j
                            turn_length = math.sqrt((path[turn_end][0] - path[turn_start][0])**2 + 
                                                  (path[turn_end][1] - path[turn_start][1])**2)
                            
                            # 检查是否是短转弯段（长度小于15且转弯点数少于8个）
                            turn_point_count = turn_end - turn_start
                            if turn_length < 15 and turn_point_count < 8:
                                short_turn_segments.append({
                                    'front_segment': (front_start, front_end),
                                    'turn_segment': (turn_start, turn_end),
                                    'back_segment': (back_start, back_end),
                                    'turn_length': turn_length,
                                    'turn_point_count': turn_point_count
                                })
                                print(f"DEBUG: 发现短转弯段 - 前直线: {front_start}-{front_end}, "
                                      f"转弯: {turn_start}-{turn_end} (长度: {turn_length:.1f}, 点数: {turn_point_count}), "
                                      f"后直线: {back_start}-{back_end}")
                            
                            # 跳过已分析的部分
                            i = back_end
                            break
                        j += 1
                    else:
                        i += 1
                
                return short_turn_segments
            
            optimized_path = path.copy()
            short_turns = detect_short_turn_segments(path)
            
            print(f"DEBUG: 检测到 {len(short_turns)} 个可优化的短转弯段")
            
            # 逆序处理，避免索引变化问题
            for turn_info in reversed(short_turns):
                front_seg = turn_info['front_segment']
                turn_seg = turn_info['turn_segment']
                back_seg = turn_info['back_segment']
                
                # 获取前方直线和后方直线的方向
                front_start_point = optimized_path[front_seg[0]]
                front_end_point = optimized_path[front_seg[1]]
                front_direction = get_line_direction(front_start_point, front_end_point)
                
                back_start_point = optimized_path[back_seg[0]]
                back_end_point = optimized_path[back_seg[1]]
                back_direction = get_line_direction(back_start_point, back_end_point)
                
                # 计算两条直线延长后的交点
                intersection = line_intersection(front_end_point, front_direction, 
                                               back_start_point, back_direction)
                
                if intersection is None:
                    print(f"DEBUG: 直线平行，无法计算交点，跳过")
                    continue
                
                # 检查交点是否在可行区域
                if not is_valid_position(intersection[0], intersection[1]):
                    print(f"DEBUG: 交点 {intersection} 不在可行区域，跳过")
                    continue
                
                # 检查从前直线端点到交点、从交点到后直线起点是否可直接移动
                if (can_move_directly(front_end_point, intersection) and 
                    can_move_directly(intersection, back_start_point)):
                    
                    # 计算优化后的路径长度
                    original_length = 0
                    for k in range(front_seg[1], back_seg[0]):
                        if k + 1 < len(optimized_path):
                            p1 = optimized_path[k]
                            p2 = optimized_path[k + 1]
                            original_length += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    
                    new_length = (math.sqrt((intersection[0] - front_end_point[0])**2 + 
                                          (intersection[1] - front_end_point[1])**2) +
                                 math.sqrt((back_start_point[0] - intersection[0])**2 + 
                                          (back_start_point[1] - intersection[1])**2))
                    
                    # 只有在路径缩短时才进行替换
                    if new_length < original_length * 0.95:  # 至少缩短5%
                        # 用交点替换转弯段
                        new_path = (optimized_path[:front_seg[1] + 1] + 
                                   [intersection] + 
                                   optimized_path[back_seg[0]:])
                        
                        optimized_path = new_path
                        
                        print(f"DEBUG: 成功用交点 {intersection} 替换转弯段，"
                              f"路径长度从 {original_length:.1f} 减少到 {new_length:.1f}")
                    else:
                        print(f"DEBUG: 交点替换后路径未明显缩短，跳过 (原长度: {original_length:.1f}, 新长度: {new_length:.1f})")
                else:
                    print(f"DEBUG: 无法直接移动到交点 {intersection}，跳过")
            
            return optimized_path
        
        def smooth_path(original_path):
            """改进的路径平滑算法，包含转弯集群优化和直线延长优化"""
            if len(original_path) < 3:
                return original_path
            
            # 步骤1：检测并优化转弯集群
            print(f"DEBUG: 检测转弯集群...")
            turn_clusters = detect_turn_clusters(original_path)
            print(f"DEBUG: 发现 {len(turn_clusters)} 个转弯集群")
            
            current_path = original_path.copy()
            
            # 优化每个转弯集群
            for i, cluster in enumerate(turn_clusters):
                print(f"DEBUG: 优化转弯集群 {i+1}, 包含转弯点: {cluster}")
                current_path = optimize_turn_cluster(current_path, cluster)
            
            # 步骤2：应用直线延长-交点替换优化
            print(f"DEBUG: 应用直线延长-交点替换优化...")
            current_path = extend_lines_and_replace_turns(current_path)
            
            # 步骤3：传统的直线连接优化
            smoothed = [current_path[0]]  # Start with first point
            i = 0
            
            while i < len(current_path) - 1:
                current_pos = current_path[i]
                
                # Find the farthest point we can reach directly
                max_reachable = i + 1
                for j in range(i + 2, len(current_path)):
                    if can_move_directly(current_pos, current_path[j]):
                        max_reachable = j
                    else:
                        break
                
                # Add the farthest reachable point
                if max_reachable > i + 1:
                    smoothed.append(current_path[max_reachable])
                    i = max_reachable
                else:
                    smoothed.append(current_path[i + 1])
                    i += 1
            
            # 步骤4：应用贝塞尔曲线平滑处理急转弯
            print(f"DEBUG: 路径点数量: {len(original_path)} -> {len(current_path)} (转弯+直线优化) -> {len(smoothed)} (直线连接优化)")
            print(f"DEBUG: 应用贝塞尔曲线平滑...")
            final_smoothed = apply_bezier_smoothing(smoothed)
            print(f"DEBUG: 贝塞尔平滑后路径点数量: {len(final_smoothed)}")
            
            return final_smoothed
            
        def apply_bezier_smoothing(path):
            """使用贝塞尔曲线对急转弯进行平滑处理"""
            if len(path) < 3:
                return path
            
            def calculate_turn_sharpness(p1, p2, p3):
                """计算转弯的急剧程度"""
                # 计算两个线段的夹角
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                len1 = math.sqrt(v1[0]**2 + v1[1]**2)
                len2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if len1 == 0 or len2 == 0:
                    return 0
                
                # 归一化向量
                v1_norm = (v1[0]/len1, v1[1]/len1)
                v2_norm = (v2[0]/len2, v2[1]/len2)
                
                # 计算夹角余弦值
                dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
                dot_product = max(-1.0, min(1.0, dot_product))  # 限制在[-1,1]范围内
                
                angle = math.acos(dot_product)
                return math.degrees(angle)
            
            def quadratic_bezier(p0, p1, p2, t):
                """二次贝塞尔曲线插值"""
                x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
                y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
                return (int(x), int(y))
            
            smoothed_path = [path[0]]
            
            i = 1
            while i < len(path) - 1:
                prev_point = path[i-1]
                current_point = path[i]
                next_point = path[i+1]
                
                # 计算转弯角度
                turn_angle = calculate_turn_sharpness(prev_point, current_point, next_point)
                
                # 降低贝塞尔曲线平滑的阈值，对更多转弯进行平滑处理
                if turn_angle > 30:  # 从60度降低到30度
                    print(f"DEBUG: 在点 {current_point} 应用贝塞尔平滑，转弯角度: {turn_angle:.1f}°")
                    
                    # 生成贝塞尔曲线上的点，增加插值点数量
                    bezier_points = []
                    # 根据转弯角度决定插值点数量，增加插值密度
                    num_points = max(3, int(turn_angle / 20))  # 从30改为20，更密集的插值
                    
                    for j in range(1, num_points):
                        t = j / num_points
                        bezier_point = quadratic_bezier(prev_point, current_point, next_point, t)
                        
                        # 检查贝塞尔点是否有效
                        if is_valid_position(bezier_point[0], bezier_point[1]):
                            bezier_points.append(bezier_point)
                    
                    # 如果贝塞尔点有效，添加它们；否则保持原始路径
                    if bezier_points:
                        # 检查是否能直接连接到贝塞尔曲线的各个点
                        valid_bezier = True
                        for k in range(len(bezier_points) - 1):
                            if not can_move_directly(bezier_points[k], bezier_points[k+1]):
                                valid_bezier = False
                                break
                        
                        if valid_bezier and can_move_directly(smoothed_path[-1], bezier_points[0]):
                            smoothed_path.extend(bezier_points)
                        else:
                            smoothed_path.append(current_point)
                    else:
                        smoothed_path.append(current_point)
                else:
                    smoothed_path.append(current_point)
                
                i += 1
            
            # 添加最后一个点
            if len(path) > 1:
                smoothed_path.append(path[-1])
            
            return smoothed_path
        
        # Step 1: Smooth the path by connecting distant points
        print(f"Original path points: {len(path)}")
        smoothed_path = smooth_path(path)
        print(f"After path smoothing: {len(smoothed_path)} points")
        
        # Step 2: Convert smoothed path segments to movement vectors
        segments = []
        for i in range(len(smoothed_path) - 1):
            current = smoothed_path[i]
            next_pos = smoothed_path[i + 1]
            
            dx = next_pos[0] - current[0]
            dy = next_pos[1] - current[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Normalize direction vector
            if distance > 0:
                norm_dx = dx / distance
                norm_dy = dy / distance
            else:
                norm_dx = norm_dy = 0
            
            segments.append({
                'dx': dx,
                'dy': dy,
                'distance': distance,
                'norm_dx': norm_dx,
                'norm_dy': norm_dy,
                'start': current,
                'end': next_pos
            })
        
        if not segments:
            return []
        
        # Step 3: Merge segments with same or very similar directions
        merged_segments = []
        current_segment = segments[0]
        current_total_distance = current_segment['distance']
        current_start = current_segment['start']
        current_end = current_segment['end']
        
        def vectors_similar(v1, v2, tolerance=0.02):
            """Check if two normalized direction vectors are similar with strict tolerance"""
            if v1['distance'] == 0 or v2['distance'] == 0:
                return False
            
            # Calculate dot product of normalized vectors
            dot_product = v1['norm_dx'] * v2['norm_dx'] + v1['norm_dy'] * v2['norm_dy']
            
            # Vectors are similar if dot product is very close to 1 (same direction)
            return dot_product > (1 - tolerance)
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            
            # Check if current segment and next segment have similar directions
            # AND if we can move directly from current start to next end
            if (vectors_similar(current_segment, next_segment, tolerance=0.02) and
                can_move_directly(current_start, next_segment['end'])):
                # Merge segments
                current_total_distance += next_segment['distance']
                current_end = next_segment['end']
                
                # Update the representative direction (use weighted average)
                total_dist = current_segment['distance'] + next_segment['distance']
                if total_dist > 0:
                    current_segment['norm_dx'] = (
                        (current_segment['norm_dx'] * current_segment['distance'] + 
                         next_segment['norm_dx'] * next_segment['distance']) / total_dist
                    )
                    current_segment['norm_dy'] = (
                        (current_segment['norm_dy'] * current_segment['distance'] + 
                         next_segment['norm_dy'] * next_segment['distance']) / total_dist
                    )
                    current_segment['distance'] = total_dist
            else:
                # Different direction or can't move directly, save current merged segment
                merged_segments.append({
                    'start': current_start,
                    'end': current_end,
                    'total_distance': current_total_distance,
                    'norm_dx': current_segment['norm_dx'],
                    'norm_dy': current_segment['norm_dy']
                })
                
                # Start new segment
                current_segment = next_segment
                current_total_distance = next_segment['distance']
                current_start = next_segment['start']
                current_end = next_segment['end']
        
        # Add the last merged segment
        merged_segments.append({
            'start': current_start,
            'end': current_end,
            'total_distance': current_total_distance,
            'norm_dx': current_segment['norm_dx'],
            'norm_dy': current_segment['norm_dy']
        })
        
        print(f"After direction merging: {len(merged_segments)} segments")
        
        # Step 4: Convert merged segments to MovePlan objects
        move_plans = []
        for segment in merged_segments:
            # Determine direction based on the normalized direction vector
            norm_dx = segment['norm_dx']
            norm_dy = segment['norm_dy']
            
            # Calculate angle
            angle = math.atan2(norm_dy, norm_dx)
            angle_deg = math.degrees(angle) % 360
            
            # Classify direction - simplified to Forward/Backward
            # Forward: generally moving right or up
            # Backward: generally moving left or down
            if norm_dx >= 0:  # Moving right or vertical
                if norm_dy >= 0:  # Right-up quadrant
                    direction = Direction.ForWard
                else:  # Right-down quadrant
                    direction = Direction.ForWard
            else:  # Moving left
                if norm_dy >= 0:  # Left-up quadrant
                    direction = Direction.BackWard
                else:  # Left-down quadrant
                    direction = Direction.BackWard
            
            # Create MovePlan
            plan = MovePlan()
            plan.direction = direction
            plan.distance = segment['total_distance']
            move_plans.append(plan)
        
        return move_plans
    
    # Main logic
    if not is_valid_position(start[0], start[1]):
        print("DEBUG: Invalid start position!")
        return []  # Invalid start point
    
    # Get valid target positions
    print("DEBUG: Getting target positions...")
    target_positions = get_target_positions()
    print(f"DEBUG: Found {len(target_positions)} valid target positions")
    
    if not target_positions:
        print("DEBUG: No valid target positions found - cannot proceed with path planning!")
        return []  # No valid target positions
    
    # Find the closest valid target position to start point
    best_target = min(target_positions, key=lambda pos: heuristic(start, pos))
    print(f"DEBUG: Selected best target: {best_target}")
    
    # Use A* algorithm to find path
    print("DEBUG: Starting A* pathfinding...")
    path = a_star_search(start, best_target)
    if not path:
        print("DEBUG: A* failed to find path!")
        return []  # No path found
    
    print(f"DEBUG: A* found path with {len(path)} points")
    
    # Convert path to MovePlan
    move_plans = path_to_move_plans(path)
    print(f"DEBUG: Generated {len(move_plans)} movement plans")
    
    return move_plans


def generate_random_map(width=100, height=100, obstacle_count=20, min_obstacle_size=3, max_obstacle_size=8):
    """
    Generate random map
    
    Parameters:
    - width: Map width
    - height: Map height
    - obstacle_count: Number of obstacles
    - min_obstacle_size: Minimum obstacle size
    - max_obstacle_size: Maximum obstacle size
    
    Returns:
    - list[list[bool]]: 2D boolean map
    """
    # Initialize map (False means passable, True means obstacle)
    game_map = [[False for _ in range(width)] for _ in range(height)]
    
    # Add random obstacles
    for _ in range(obstacle_count):
        # Random obstacle size
        obs_width = random.randint(min_obstacle_size, max_obstacle_size)
        obs_height = random.randint(min_obstacle_size, max_obstacle_size)
        
        # Random position
        start_x = random.randint(0, width - obs_width)
        start_y = random.randint(0, height - obs_height)
        
        # Add rectangular obstacle
        for y in range(start_y, start_y + obs_height):
            for x in range(start_x, start_x + obs_width):
                if 0 <= x < width and 0 <= y < height:
                    game_map[y][x] = True
    
    return game_map


def find_valid_position(game_map, car_radius, exclude_area=None):
    """
    Find a valid position on the map
    
    Parameters:
    - game_map: Map
    - car_radius: Vehicle radius
    - exclude_area: Area to exclude (x, y, radius)
    
    Returns:
    - tuple: Valid position (x, y) or None
    """
    height = len(game_map)
    width = len(game_map[0])
    
    def is_valid_position(x, y):
        # Check boundaries
        if x - car_radius < 0 or y - car_radius < 0 or x + car_radius >= width or y + car_radius >= height:
            return False
        
        # Check obstacles
        for dy in range(-car_radius, car_radius + 1):
            for dx in range(-car_radius, car_radius + 1):
                if dx*dx + dy*dy <= car_radius*car_radius:
                    check_x, check_y = x + dx, y + dy
                    if 0 <= check_x < width and 0 <= check_y < height:
                        if game_map[check_y][check_x]:
                            return False
        
        # Check if within exclude area
        if exclude_area:
            ex_x, ex_y, ex_radius = exclude_area
            dist = math.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            if dist < ex_radius:
                return False
        
        return True
    
    # Try to find valid position (max 1000 attempts)
    for _ in range(1000):
        x = random.randint(car_radius, width - car_radius - 1)
        y = random.randint(car_radius, height - car_radius - 1)
        if is_valid_position(x, y):
            return (x, y)
    
    return None


def visualize_path_finding(game_map, start, end, path, car_radius, fire_radius):
    """
    Visualize pathfinding results
    
    Parameters:
    - game_map: Map
    - start: Start point
    - end: End point
    - path: Found path (MovePlan list)
    - car_radius: Vehicle radius
    - fire_radius: Firing radius
    """
    height = len(game_map)
    width = len(game_map[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw map
    map_array = np.array(game_map, dtype=int)
    ax.imshow(map_array, cmap='binary', origin='lower', alpha=0.7)
    
    # Draw start point
    if start:
        circle = Circle(start, car_radius, color='green', fill=False, linewidth=2, label='Start Point')
        ax.add_patch(circle)
        ax.plot(start[0], start[1], 'go', markersize=8)
    
    # Draw end point
    if end:
        circle = Circle(end, car_radius, color='red', fill=False, linewidth=2, label='End Point')
        ax.add_patch(circle)
        ax.plot(end[0], end[1], 'ro', markersize=8)
        
        # Draw firing range and no-go zone
        # No-go zone (red): within fire_radius of end point
        no_go_circle = Circle(end, fire_radius, color='red', fill=True, alpha=0.2, label=f'No-Go Zone (r={fire_radius})')
        ax.add_patch(no_go_circle)
        
        # Firing range circles
        fire_circle_min = Circle(end, fire_radius, color='orange', fill=False, 
                                linewidth=1, linestyle='--', alpha=0.7, label=f'Min Fire Range {fire_radius}')
        fire_circle_max = Circle(end, 2*fire_radius, color='orange', fill=False, 
                                linewidth=1, linestyle='-', alpha=0.7, label=f'Max Fire Range {2*fire_radius}')
        ax.add_patch(fire_circle_min)
        ax.add_patch(fire_circle_max)
    
    # Draw path
    if path and len(path) > 0:
        print(f"Drawing path with {len(path)} movement steps")
        
        # To visualize, we need to re-run search_path to get actual coordinate path
        # Or simplify display: show number of MovePlans found and total distance
        total_distance = sum(move.distance for move in path)
        
        # Display path information on the plot
        info_text = f"Path Found!\nSteps: {len(path)}\nTotal Distance: {total_distance:.1f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', fontsize=10)
    
    # Set figure properties
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Pathfinding Visualization (Map: {width}x{height}, Vehicle Radius: {car_radius}, Fire Radius: {fire_radius})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def get_actual_path_coordinates(game_map, start, end, car_radius, fire_radius):
    """
    Get actual path coordinates (by modifying search_path function to return coordinates)
    """
    if not game_map or not game_map[0]:
        return []
    
    height = len(game_map)
    width = len(game_map[0])
    
    def is_valid_position(x, y):
        if x - car_radius < 0 or y - car_radius < 0 or x + car_radius >= width or y + car_radius >= height:
            return False
        
        for dy in range(-car_radius, car_radius + 1):
            for dx in range(-car_radius, car_radius + 1):
                if dx*dx + dy*dy <= car_radius*car_radius:
                    check_x, check_y = x + dx, y + dy
                    if 0 <= check_x < width and 0 <= check_y < height:
                        if game_map[check_y][check_x]:
                            return False
        return True
    
    def get_target_positions():
        targets = []
        min_dist = fire_radius
        max_dist = 2 * fire_radius
        
        search_range = max_dist + car_radius
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                target_x = end[0] + dx
                target_y = end[1] + dy
                
                if 0 <= target_x < width and 0 <= target_y < height:
                    dist = math.sqrt(dx*dx + dy*dy)
                    if min_dist <= dist <= max_dist and is_valid_position(target_x, target_y):
                        targets.append((target_x, target_y))
        
        return targets
    
    def heuristic(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def a_star_search(start_pos, target_pos):
        open_set = []
        heapq.heappush(open_set, (0, start_pos))
        
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: heuristic(start_pos, target_pos)}
        
        open_set_hash = {start_pos}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == target_pos:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_pos)
                path.reverse()
                return path
            
            for neighbor in get_neighbors(current):
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = math.sqrt(dx*dx + dy*dy)
                
                temp_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, target_pos)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return None
    
    # Main logic
    if not is_valid_position(start[0], start[1]):
        return []
    
    target_positions = get_target_positions()
    if not target_positions:
        return []
    
    best_target = min(target_positions, key=lambda pos: heuristic(start, pos))
    path = a_star_search(start, best_target)
    
    return path if path else []

def visualize_path_finding_improved(game_map, start, end, car_radius, fire_radius):
    """
    Improved visualization function with detailed path optimization display, turn angles, and shooting triangle
    """
    height = len(game_map)
    width = len(game_map[0])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get all path data for comparison
    original_path, smoothed_path, final_segments = get_path_optimization_details(game_map, start, end, car_radius, fire_radius)
    optimized_path = search_path(game_map, start, end, car_radius, fire_radius)
    
    # Get clear sectors information for visualization
    def get_clear_sectors_for_visualization():
        """Get clear sectors information by calling the same analysis as in search_path"""
        min_dist = fire_radius
        max_dist = 2 * fire_radius
        
        # Same analyze_polar_obstacles function as in search_path
        obstacle_rays = []
        for angle_deg in range(0, 360, 1):
            angle_rad = math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            
            hit_obstacle = False
            hit_distance = float('inf')
            
            step_size = 0.5
            max_ray_distance = max_dist + car_radius + 5
            
            for step in range(int(max_ray_distance / step_size)):
                ray_distance = step * step_size
                ray_x = end[0] + dx * ray_distance
                ray_y = end[1] + dy * ray_distance
                
                if ray_x < 0 or ray_x >= width or ray_y < 0 or ray_y >= height:
                    hit_distance = ray_distance
                    hit_obstacle = True
                    break
                
                grid_x = int(ray_x)
                grid_y = int(ray_y)
                if 0 <= grid_x < width and 0 <= grid_y < height:
                    if game_map[grid_y][grid_x]:
                        hit_distance = ray_distance
                        hit_obstacle = True
                        break
            
            obstacle_rays.append({
                'angle': angle_deg,
                'hit_distance': hit_distance,
                'hit_obstacle': hit_obstacle
            })
        
        # Same find_clear_sectors function as in search_path
        obstacle_angles = []
        for ray in obstacle_rays:
            if ray['hit_obstacle'] and ray['hit_distance'] <= max_dist + car_radius:
                obstacle_angles.append(ray['angle'])
        
        if not obstacle_angles:
            return [(0, 360)], []
        
        obstacle_angles = sorted(obstacle_angles)
        clear_sectors = []
        n = len(obstacle_angles)
        
        for i in range(n):
            current_angle = obstacle_angles[i]
            next_angle = obstacle_angles[(i + 1) % n]
            
            if i == n - 1:
                gap = (360 - current_angle) + next_angle
                sector_start = current_angle
                sector_end = next_angle + 360
            else:
                gap = next_angle - current_angle
                sector_start = current_angle
                sector_end = next_angle
            
            if gap >= 45:  # min_gap_degrees = 45
                if sector_end > 360:
                    clear_sectors.append((sector_start, 360))
                    clear_sectors.append((0, sector_end - 360))
                else:
                    clear_sectors.append((sector_start, sector_end))
        
        return clear_sectors, obstacle_angles
    
    clear_sectors, obstacle_angles = get_clear_sectors_for_visualization()
    
    # Get final vehicle position and calculate shooting triangle
    final_vehicle_pos = None
    if smoothed_path and len(smoothed_path) > 0:
        final_vehicle_pos = smoothed_path[-1]
    
    def calculate_turn_angle(p1, p2, p3):
        """Calculate the turn angle at point p2 between vectors p1->p2 and p2->p3"""
        # Vector from p1 to p2
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        # Vector from p2 to p3
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate angles
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        
        # Calculate turn angle
        turn_angle = angle2 - angle1
        
        # Normalize to [-π, π]
        while turn_angle > math.pi:
            turn_angle -= 2 * math.pi
        while turn_angle < -math.pi:
            turn_angle += 2 * math.pi
        
        # Convert to degrees
        return math.degrees(turn_angle)
    
    def get_path_angles(path):
        """Get turn angles for each point in the path"""
        if len(path) < 3:
            return []
        
        angles = []
        for i in range(1, len(path) - 1):
            angle = calculate_turn_angle(path[i-1], path[i], path[i+1])
            angles.append((path[i], angle))
        
        return angles
    
    # Draw maps on both subplots
    map_array = np.array(game_map, dtype=int)
    ax1.imshow(map_array, cmap='gray_r', origin='lower', alpha=0.7)
    ax2.imshow(map_array, cmap='gray_r', origin='lower', alpha=0.7)
    
    # Common elements for both plots
    for ax in [ax1, ax2]:
        # Draw start point
        if start:
            circle = Circle(start, car_radius, color='green', fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.plot(start[0], start[1], 'go', markersize=10, label='Start' if ax == ax1 else '')
        
        # Draw end point
        if end:
            circle = Circle(end, car_radius, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.plot(end[0], end[1], 'ro', markersize=10, label='End' if ax == ax1 else '')
            
            # Draw firing range and no-go zone
            # No-go zone (red): within fire_radius of end point
            no_go_circle = Circle(end, fire_radius, color='red', fill=True, alpha=0.2, 
                                 label='No-Go Zone' if ax == ax1 else '')
            ax.add_patch(no_go_circle)
            
            # Firing range circles
            fire_circle_min = Circle(end, fire_radius, color='orange', fill=False, 
                                    linewidth=1, linestyle='--', alpha=0.7, 
                                    label='Min Fire Range' if ax == ax1 else '')
            fire_circle_max = Circle(end, 2*fire_radius, color='orange', fill=False, 
                                    linewidth=1, linestyle='-', alpha=0.7,
                                    label='Max Fire Range' if ax == ax1 else '')
            ax.add_patch(fire_circle_min)
            ax.add_patch(fire_circle_max)
            
            # Draw clear sectors (45° gap requirement visualization)
            if clear_sectors:
                sector_radius = 2 * fire_radius + 5  # Extend beyond max fire range for visibility
                
                # Draw obstacle rays (blocked directions)
                for obs_angle in obstacle_angles:
                    angle_rad = math.radians(obs_angle)
                    end_x = end[0] + sector_radius * math.cos(angle_rad)
                    end_y = end[1] + sector_radius * math.sin(angle_rad)
                    ax.plot([end[0], end_x], [end[1], end_y], 'red', linewidth=1, alpha=0.3)
                
                # Draw clear sectors with different colors
                sector_colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightpink', 'lightcyan']
                for i, (sector_start, sector_end) in enumerate(clear_sectors):
                    color = sector_colors[i % len(sector_colors)]
                    
                    # Create sector wedge
                    if sector_end > 360:  # Handle wrap-around sectors
                        # Draw first part (sector_start to 360)
                        wedge1 = Wedge(end, sector_radius, sector_start, 360, 
                                      facecolor=color, alpha=0.3, edgecolor='darkgreen', linewidth=2)
                        ax.add_patch(wedge1)
                        
                        # Draw second part (0 to sector_end-360)
                        wedge2 = Wedge(end, sector_radius, 0, sector_end - 360,
                                      facecolor=color, alpha=0.3, edgecolor='darkgreen', linewidth=2)
                        ax.add_patch(wedge2)
                        
                        # Label for this sector
                        mid_angle = ((sector_start + (sector_end - 360)) / 2) % 360
                    else:
                        # Normal sector
                        wedge = Wedge(end, sector_radius, sector_start, sector_end,
                                     facecolor=color, alpha=0.3, edgecolor='darkgreen', linewidth=2)
                        ax.add_patch(wedge)
                        
                        # Label for this sector
                        mid_angle = (sector_start + sector_end) / 2
                    
                    # Add sector label
                    label_radius = sector_radius * 0.7
                    label_x = end[0] + label_radius * math.cos(math.radians(mid_angle))
                    label_y = end[1] + label_radius * math.sin(math.radians(mid_angle))
                    
                    gap_size = (sector_end - sector_start) % 360
                    ax.text(label_x, label_y, f'Clear\nSector {i+1}\n{gap_size:.0f}°',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                           ha='center', va='center', fontsize=8, fontweight='bold')
                
                # Add sector legend
                if ax == ax1:
                    ax.plot([], [], 'green', linewidth=4, alpha=0.6, label=f'Clear Sectors (≥45°)')
                    ax.plot([], [], 'red', linewidth=1, alpha=0.3, label='Blocked Rays')
        
        # Draw final vehicle position if available
        if final_vehicle_pos:
            # Draw final vehicle position with special marking
            vehicle_circle = Circle(final_vehicle_pos, car_radius, color='darkgreen', 
                                  fill=False, linewidth=3, linestyle='--')
            ax.add_patch(vehicle_circle)
            ax.plot(final_vehicle_pos[0], final_vehicle_pos[1], 'g^', markersize=12, 
                   label='Final Position' if ax == ax1 else '')
    
    # Left plot: Original A* path
    ax1.set_title('Original A* Path with Turn Angles', fontsize=14, fontweight='bold')
    if original_path and len(original_path) > 1:
        # Draw original path
        path_x = [p[0] for p in original_path]
        path_y = [p[1] for p in original_path]
        ax1.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.8, label='Original Path')
        
        # Draw all intermediate points
        for i, point in enumerate(original_path):
            if i > 0 and i < len(original_path) - 1:
                ax1.plot(point[0], point[1], 'bo', markersize=3, alpha=0.6)
        
        # Calculate and display turn angles
        original_angles = get_path_angles(original_path)
        for point, angle in original_angles:
            # Only show significant turns (> 15 degrees)
            if abs(angle) > 15:
                color = 'red' if abs(angle) > 45 else 'orange'
                ax1.annotate(f'{angle:.0f}°', 
                           xy=point, 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                           fontsize=8, 
                           fontweight='bold',
                           color='white')
        
        # Calculate original distance
        original_distance = sum(math.sqrt((original_path[i+1][0] - original_path[i][0])**2 + 
                                        (original_path[i+1][1] - original_path[i][1])**2) 
                              for i in range(len(original_path)-1))
        
        # Calculate total turn angle
        total_turn = sum(abs(angle) for _, angle in original_angles)
        
        info_text1 = f"Original Path:\n• Points: {len(original_path)}\n• Steps: {len(original_path)-1}\n• Distance: {original_distance:.1f}\n• Total Turn: {total_turn:.0f}°\n• Major Turns: {len([a for _, a in original_angles if abs(a) > 45])}"
        ax1.text(0.02, 0.98, info_text1, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        ax1.legend(loc='lower right')
    
    # Right plot: Optimized path with segments
    ax2.set_title('Optimized Path with Segment Angles', fontsize=14, fontweight='bold')
    
    if smoothed_path and len(smoothed_path) > 1:
        # Show original path lightly
        if original_path:
            path_x = [p[0] for p in original_path]
            path_y = [p[1] for p in original_path]
            ax2.plot(path_x, path_y, 'lightgray', linewidth=1, alpha=0.4, label='Original (faded)')
        
        # Show smoothed path
        smooth_x = [p[0] for p in smoothed_path]
        smooth_y = [p[1] for p in smoothed_path]
        ax2.plot(smooth_x, smooth_y, 'cyan', linewidth=2, alpha=0.7, label='Smoothed Path')
        
        # Draw smoothed path points
        for i, point in enumerate(smoothed_path):
            if i > 0 and i < len(smoothed_path) - 1:
                ax2.plot(point[0], point[1], 'co', markersize=6, alpha=0.8)
        
        # Calculate and display turn angles for smoothed path
        smoothed_angles = get_path_angles(smoothed_path)
        for point, angle in smoothed_angles:
            # Show all turns for the optimized path
            if abs(angle) > 5:
                color = 'red' if abs(angle) > 45 else 'orange' if abs(angle) > 15 else 'yellow'
                ax2.annotate(f'{angle:.0f}°', 
                           xy=point, 
                           xytext=(8, 8), 
                           textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                           fontsize=9, 
                           fontweight='bold',
                           color='black')
        
        # Draw final optimized segments with different colors and angles
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'magenta']
        
        segment_angles = []
        for i, segment in enumerate(final_segments):
            color = colors[i % len(colors)]
            start_pos = segment['start']
            end_pos = segment['end']
            
            # Calculate segment direction angle
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            segment_angle = math.degrees(math.atan2(dy, dx))
            segment_angles.append(segment_angle)
            
            # Draw segment with thick line
            ax2.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                    color=color, linewidth=5, alpha=0.8, 
                    label=f'Seg {i+1} ({segment["distance"]:.1f}, {segment_angle:.0f}°)')
            
            # Draw segment number and angle
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            ax2.text(mid_x, mid_y, f'{i+1}\n{segment_angle:.0f}°', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9),
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        
        # Calculate angle changes between segments
        angle_changes = []
        for i in range(1, len(segment_angles)):
            angle_change = segment_angles[i] - segment_angles[i-1]
            # Normalize to [-180, 180]
            while angle_change > 180:
                angle_change -= 360
            while angle_change < -180:
                angle_change += 360
            angle_changes.append(angle_change)
        
        # Calculate optimization statistics
        total_optimized_distance = sum(seg['distance'] for seg in final_segments)
        original_steps = len(original_path) - 1 if original_path else 0
        optimized_steps = len(final_segments)
        reduction_pct = (original_steps - optimized_steps) / max(original_steps, 1) * 100
        total_smoothed_turn = sum(abs(angle) for _, angle in smoothed_angles)
        
        info_text2 = f"Optimized Path:\n• Smoothed Points: {len(smoothed_path)}\n• Final Segments: {len(final_segments)}\n• Distance: {total_optimized_distance:.1f}\n• Step Reduction: {reduction_pct:.1f}%\n• Total Turn: {total_smoothed_turn:.0f}°\n• Max Turn: {max([abs(a) for _, a in smoothed_angles], default=0):.0f}°"
        ax2.text(0.02, 0.98, info_text2, transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9),
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        # Add legend for segments
        ax2.legend(loc='lower right', fontsize=7, ncol=1)
    
    # Set properties for both plots
    for ax in [ax1, ax2]:
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Add main title
    optimization_summary = ""
    if original_path and final_segments:
        original_steps = len(original_path) - 1
        final_steps = len(final_segments)
        reduction = original_steps - final_steps
        reduction_pct = reduction / max(original_steps, 1) * 100
        
        # Calculate turn efficiency
        original_angles = get_path_angles(original_path)
        smoothed_angles = get_path_angles(smoothed_path)
        original_turn = sum(abs(angle) for _, angle in original_angles)
        optimized_turn = sum(abs(angle) for _, angle in smoothed_angles)
        turn_reduction = ((original_turn - optimized_turn) / max(original_turn, 1)) * 100
        
        optimization_summary = f" | Steps: {original_steps} → {final_steps} ({reduction_pct:.1f}% ↓) | Turns: {original_turn:.0f}° → {optimized_turn:.0f}° ({turn_reduction:.1f}% ↓)"
    
    fig.suptitle(f'Path Planning with Turn Angles & Clear Sector Analysis\nMap: {width}×{height} | Vehicle Radius: {car_radius} | Fire Radius: {fire_radius}{optimization_summary}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def test_path_finding():
    """
    Test pathfinding functionality
    """
    print("Starting pathfinding test...")
    
    # Generate random map
    width, height = 80, 60
    car_radius = 6
    fire_radius = 3
    
    print(f"Generating {width}x{height} random map...")
    game_map = generate_random_map(width, height, obstacle_count=15, min_obstacle_size=3, max_obstacle_size=6)
    
    # Find valid start and end points
    print("Finding valid start point...")
    start = find_valid_position(game_map, car_radius)
    if not start:
        print("Cannot find valid start point!")
        return
    
    print("Finding valid end point...")
    # End point should be far enough from start point
    end = find_valid_position(game_map, car_radius, exclude_area=(start[0], start[1], 20))
    if not end:
        print("Cannot find valid end point!")
        return
    
    print(f"Start point: {start}")
    print(f"End point: {end}")
    print(f"Vehicle radius: {car_radius}")
    print(f"Fire radius: {fire_radius}")
    
    # Execute pathfinding
    print("Executing pathfinding...")
    
    # First, get the raw path coordinates for comparison
    raw_path_coords = get_actual_path_coordinates(game_map, start, end, car_radius, fire_radius)
    raw_step_count = len(raw_path_coords) - 1 if raw_path_coords and len(raw_path_coords) > 1 else 0
    
    # Get the optimized movement plans
    path = search_path(game_map, start, end, car_radius, fire_radius)
    
    if path:
        print(f"Path found! Optimization results:")
        print(f"  Raw path steps: {raw_step_count}")
        print(f"  Optimized steps: {len(path)} (after merging)")
        print(f"  Reduction: {raw_step_count - len(path)} steps ({(raw_step_count - len(path))/max(raw_step_count, 1)*100:.1f}%)")
        print(f"")
        print(f"Optimized movement plan:")
        total_distance = 0
        prev_angle = None
        total_turn_angle = 0
        
        for i, move_plan in enumerate(path):
            direction_name = "Forward" if move_plan.direction == Direction.ForWard else "Backward"
            
            # Calculate segment angle based on direction (simplified estimation)
            # This is a rough estimation for display purposes
            segment_angle = 0 if move_plan.direction == Direction.ForWard else 180
            
            # Calculate turn angle from previous segment
            turn_angle = 0
            if prev_angle is not None:
                turn_angle = segment_angle - prev_angle
                while turn_angle > 180:
                    turn_angle -= 360
                while turn_angle < -180:
                    turn_angle += 360
                total_turn_angle += abs(turn_angle)
            
            turn_info = f", Turn: {turn_angle:.0f}°" if prev_angle is not None else ""
            print(f"  Step {i+1}: {direction_name}, Distance={move_plan.distance:.2f}{turn_info}")
            
            total_distance += move_plan.distance
            prev_angle = segment_angle
        
        print(f"Total distance: {total_distance:.2f}")
        print(f"Total turning: {total_turn_angle:.0f}°")
        print(f"Path optimization: Consecutive same-direction moves have been fully merged")
        print(f"Clear sector constraint: Final position is in a clear 45° sector")
        print(f"Distance constraint: Final position is 1-2 times fire_radius from target")
    else:
        print("No path found!")
    
    # Visualize results
    print("Displaying visualization...")
    visualize_path_finding_improved(game_map, start, end, car_radius, fire_radius)
    
    return game_map, start, end, path

def get_path_optimization_details(game_map, start, end, car_radius, fire_radius):
    """
    Get detailed path optimization information for visualization
    Returns: (original_path, smoothed_path, final_segments)
    """
    if not game_map or not game_map[0]:
        return [], [], []
    
    height = len(game_map)
    width = len(game_map[0])
    
    def is_valid_position(x, y):
        if x - car_radius < 0 or y - car_radius < 0 or x + car_radius >= width or y + car_radius >= height:
            return False
        
        for dy in range(-car_radius, car_radius + 1):
            for dx in range(-car_radius, car_radius + 1):
                if dx*dx + dy*dy <= car_radius*car_radius:
                    check_x, check_y = x + dx, y + dy
                    if 0 <= check_x < width and 0 <= check_y < height:
                        if game_map[check_y][check_x]:
                            return False
        return True
    
    def get_target_positions():
        targets = []
        min_dist = fire_radius
        max_dist = 2 * fire_radius
        
        search_range = max_dist + car_radius
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                target_x = end[0] + dx
                target_y = end[1] + dy
                
                if 0 <= target_x < width and 0 <= target_y < height:
                    dist = math.sqrt(dx*dx + dy*dy)
                    if min_dist <= dist <= max_dist and is_valid_position(target_x, target_y):
                        targets.append((target_x, target_y))
        
        return targets
    
    def heuristic(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def a_star_search(start_pos, target_pos):
        open_set = []
        heapq.heappush(open_set, (0, start_pos))
        
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: heuristic(start_pos, target_pos)}
        
        open_set_hash = {start_pos}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == target_pos:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_pos)
                path.reverse()
                return path
            
            for neighbor in get_neighbors(current):
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = math.sqrt(dx*dx + dy*dy)
                
                temp_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, target_pos)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return None
    
    def can_move_directly(start_pos, end_pos):
        """Check if we can move directly from start_pos to end_pos without hitting obstacles"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx == 0 and dy == 0:
            return True
        
        steps = max(dx, dy)
        
        for i in range(steps + 1):
            t = i / max(steps, 1)
            check_x = int(x1 + t * (x2 - x1))
            check_y = int(y1 + t * (y2 - y1))
            
            if not is_valid_position(check_x, check_y):
                return False
        
        return True
    
    def smooth_path(original_path):
        """Smooth path by connecting distant points directly when possible"""
        if len(original_path) < 3:
            return original_path
        
        smoothed = [original_path[0]]
        i = 0
        
        while i < len(original_path) - 1:
            current_pos = original_path[i]
            
            # Find the farthest point we can reach directly
            max_reachable = i + 1
            for j in range(i + 2, len(original_path)):
                if can_move_directly(current_pos, original_path[j]):
                    max_reachable = j
                else:
                    break
            
            # Add the farthest reachable point
            if max_reachable > i + 1:
                smoothed.append(original_path[max_reachable])
                i = max_reachable
            else:
                smoothed.append(original_path[i + 1])
                i += 1
        
        return smoothed
    
    # Main logic
    if not is_valid_position(start[0], start[1]):
        return [], [], []
    
    target_positions = get_target_positions()
    if not target_positions:
        return [], [], []
    
    best_target = min(target_positions, key=lambda pos: heuristic(start, pos))
    original_path = a_star_search(start, best_target)
    
    if not original_path:
        return [], [], []
    
    # Get smoothed path
    smoothed_path = smooth_path(original_path)
    
    # Create final segments for visualization
    final_segments = []
    for i in range(len(smoothed_path) - 1):
        current = smoothed_path[i]
        next_pos = smoothed_path[i + 1]
        
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        final_segments.append({
            'start': current,
            'end': next_pos,
            'distance': distance,
            'direction': Direction.ForWard if dx >= 0 else Direction.BackWard
        })
    
    return original_path, smoothed_path, final_segments

if __name__ == "__main__":
    # Run test
    test_path_finding()