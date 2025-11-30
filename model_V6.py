import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_similarity(radar_patches, mde_out_patches):
    """
    Compute similarity between radar patches and vp_ground_truth patches
    Only compares patches at the same width position (w=0 with w=0, etc.)
    
    Args:
        radar_patches: [W, B, C, H, 1]
        vp_gt_patches: [W, B, C, H, 1]
    
    Returns:
        similarity_scores: [W, B] - similarity at each position
    """
    W, B, C, H, _ = radar_patches.shape
    
    # Reshape for easier computation: [W, B, H]
    radar_flat = radar_patches.squeeze(-1).squeeze(2)  # [W, B, H]
    vp_gt_flat = mde_out_patches.squeeze(-1).squeeze(2)  # [W, B, H]
    
    # Compute similarity only at matching positions (w=0 with w=0, etc.)
    # Use negative MSE as similarity (higher is more similar)
    similarity_scores = -torch.mean((radar_flat - vp_gt_flat) ** 2, dim=2)  # [W, B]
    
    return similarity_scores

def create_radar_gt(radar_patches, mde_out_patches, similarity_scores):
    """
    Create radar ground truth by relocating radar valid values to correct elevation positions
    based on vp_gt depth similarity
    
    Args:
        radar_patches: [W, B, C, H, 1]
        vp_gt_patches: [W, B, C, H, 1]
        similarity_scores: [W, B] - similarity at each position (not used currently)
    
    Returns:
        radar_gt: [B, C, H, W] - radar values relocated to correct elevation positions
    """
    W, B, C, H, _ = radar_patches.shape
    
    # Initialize output
    radar_gt = torch.zeros(B, C, H, W, device=radar_patches.device)
    
    # For each width position and batch
    for w in range(W):
        for b in range(B):
            # Get radar and vp_gt values at this position [C, H]
            radar_col = radar_patches[w, b, :, :, 0]  # [C, H]
            mde_out_col = mde_out_patches[w, b, :, :, 0]  # [C, H]
            
            # Find valid radar positions (non-zero depth values)
            radar_valid_mask = (radar_col != 0)  # [C, H]
            
            if radar_valid_mask.sum() == 0:
                continue  # No valid radar points in this column
            
            # Get valid radar depth values and their y positions
            radar_valid_y_indices = torch.where(radar_valid_mask[0])[0]  # y positions with valid radar
            radar_valid_depths = radar_col[0, radar_valid_y_indices]  # depth values at those positions
            
            # For each valid radar point, find the best matching y position in vp_gt
            for idx, (y_radar, depth_radar) in enumerate(zip(radar_valid_y_indices, radar_valid_depths)):
                # Find vp_gt positions with valid depth
                vp_gt_valid_mask = (mde_out_col[0] != 0)
                
                if vp_gt_valid_mask.sum() == 0:
                    # No valid vp_gt, keep radar at original position
                    # Find nearest empty position around original y_radar
                    final_y = y_radar
                    if radar_gt[b, 0, y_radar, w] != 0:
                        # Search for nearest empty position
                        max_search_range = H  # Search entire height if needed
                        found = False
                        for offset in range(1, max_search_range):
                            # Try positive offset first, then negative
                            for dy in [offset, -offset]:
                                new_y = y_radar + dy
                                if 0 <= new_y < H and radar_gt[b, 0, new_y, w] == 0:
                                    final_y = new_y
                                    found = True
                                    break
                            if found:
                                break
                    radar_gt[b, 0, final_y, w] = depth_radar
                    continue
                
                # Get vp_gt valid positions and depths
                vp_gt_valid_y_indices = torch.where(vp_gt_valid_mask)[0]
                vp_gt_valid_depths = mde_out_col[0, vp_gt_valid_y_indices]
                
                # Find the closest depth match in vp_gt
                depth_differences = torch.abs(vp_gt_valid_depths - depth_radar)
                closest_idx = torch.argmin(depth_differences)
                best_y_position = vp_gt_valid_y_indices[closest_idx]
                
                # Check if the position is already occupied
                final_y_position = best_y_position
                if radar_gt[b, 0, best_y_position, w] != 0:
                    # Position is occupied, find nearest empty position
                    max_search_range = H  # Search entire height if needed
                    found = False
                    for offset in range(1, max_search_range):
                        # Try positive offset first, then negative
                        for dy in [offset, -offset]:
                            new_y = best_y_position + dy
                            if 0 <= new_y < H and radar_gt[b, 0, new_y, w] == 0:
                                final_y_position = new_y
                                found = True
                                break
                        if found:
                            break
                
                # Place radar depth value at the final y position
                radar_gt[b, 0, final_y_position, w] = depth_radar
    
    return radar_gt

class RadarElevationLearner(nn.Module):
    """Radar elevation learning with cross-attention"""
    
    def __init__(self, hidden_dim=1, num_heads=1):
        super(RadarElevationLearner, self).__init__()
        
        # Multi-head cross-attention using PyTorch's MultiHeadAttention
        # embed_dim = W (width), seq_len = C*H*1
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,  # W = 1
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization for cross-attention
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        # Small residual scale to keep gradients flowing through attention
        self.attn_residual_scale = nn.Parameter(torch.tensor(1e-9), requires_grad=True)
    
    def apply_cross_attention(self, query, key, value):
        """Cross-attention using PyTorch's MultiHeadAttention"""
        # Apply multi-head cross-attention
        attended_output, attention_weights = self.cross_attention(
            query=query,
            key=key,
            value=value,
            need_weights=True,
            average_attn_weights=False  # keep per-head weights: [N, H, Q, K]
        )
        
        # Apply layer normalization with residual connection
        output = self.cross_attention_norm(query + attended_output)
        
        return output, attention_weights
    
    
    
    def gumbel_softmax_attention(self, attention_weights, valid_indices, temperature=1.0):
        """Gumbel-Softmax를 사용하여 attention 기반 매핑"""
        N, num_heads, seq_len, _ = attention_weights.shape
        
        # valid_indices에 해당하는 attention weights 추출
        # attention_weights: [N, num_heads, seq_len, seq_len]
        # valid_indices: [N, seq_len] (각 시퀀스별 valid mask)
        
        mapped_indices = []
        for n in range(N):
            batch_valid_count = valid_indices[n].sum().item()
            if batch_valid_count == 0:
                mapped_indices.append(torch.empty(0, dtype=torch.long, device=attention_weights.device))
                continue
                
            # 해당 배치의 valid 위치들
            valid_positions = valid_indices[n].nonzero().squeeze(-1)  # [T]
            
            # valid 위치들의 attention weights: [T, seq_len]
            valid_attention = attention_weights[n, 0, valid_positions, :]  # [T, seq_len]
            
            # Gumbel noise 추가
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(valid_attention) + 1e-8) + 1e-8)
            gumbel_attention = valid_attention + gumbel_noise
            
            # Softmax로 확률 분포 생성
            softmax_attention = F.softmax(gumbel_attention / temperature, dim=-1)
            
            # 가장 높은 확률의 인덱스 선택 (argmax 대신 Gumbel-Softmax)
            mapped_idx = torch.multinomial(softmax_attention, 1).squeeze(-1)  # [T]
            mapped_indices.append(mapped_idx)
        
        return mapped_indices
    
    def forward(self, radar_patches, dmde_out_patches):
        """Radar elevation learning forward pass with per-patch attention and mapping"""
        # Inputs: [W, B, C, H, 1] where C=1, H=900
        B = radar_patches.shape[1]
        W = radar_patches.shape[0]

        # 패치별 시퀀스로 변환: [B*W, 900, 1]
        radar_seq = radar_patches.permute(1, 0, 2, 3, 4).reshape(B * W, -1, 1)
        mde_seq = dmde_out_patches.permute(1, 0, 2, 3, 4).reshape(B * W, -1, 1)

        # Cross-attention (패치별 독립): attn_weights -> [N, H, 900, 900]
        attended_out, attention_weights = self.apply_cross_attention(radar_seq, mde_seq, mde_seq)

        # radar에서 valid mask 및 값: [N, 900]
        radar_seq_flat = radar_seq.squeeze(-1)
        valid_mask = (radar_seq_flat != 0)

        # ST Gumbel-Softmax 하드 매핑 (미분 가능)
        # head 평균: [N, 900, 900]
        attn_mean = attention_weights.mean(dim=1)
        temperature = 1.0
        N = radar_seq.shape[0]
        output_seq = torch.zeros_like(radar_seq)
        
        for n in range(N):
            valid_rows = valid_mask[n]
            T = int(valid_rows.sum().item())
            if T == 0:
                continue
            # logits: [T, 900]
            logits = attn_mean[n, valid_rows, :]
            # Gumbel noise
            g_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits_g = (logits + g_noise) / temperature
            p = F.softmax(logits_g, dim=-1)  # [T, 900]
            # hard one-hot via argmax
            idx = torch.argmax(p, dim=-1)
            y_hard = F.one_hot(idx, num_classes=logits.size(1)).float()
            # straight-through estimator
            y = (y_hard - p).detach() + p  # [T, 900]
            # source values for valid rows: [T]
            src_vals = radar_seq_flat[n, valid_rows]
            # aggregate to keys: [900]
            out_n = torch.matmul(y.transpose(0, 1), src_vals)
            output_seq[n, :, 0] = out_n

        # Attention residual을 valid 영역에만 적용
        # valid_mask를 사용하여 valid 영역만 residual 적용
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # [N, 900, 1]
        attended_residual = self.attn_residual_scale * attended_out
        
        # Residual을 적용하되, 원래 0이었던 위치는 0으로 유지
        # output_seq에서 0이 아닌 위치에만 residual 적용
        output_mask = (output_seq != 0).float()  # [N, 900, 1]
        combined_mask = valid_mask_expanded * output_mask  # 두 마스크의 교집합
        output_seq = output_seq + combined_mask * attended_residual

        # 이미지 형태로 복원: [N, 900, 1] -> [B, W, 900] -> [B, 1, 900, W]
        output_bw_900 = output_seq.squeeze(-1).view(B, W, -1)
        output = output_bw_900.permute(0, 2, 1).unsqueeze(1)

        return output



class DepthRefinementUNet(nn.Module):
    """UNet 구조의 Depth refinement 네트워크"""
    
    def __init__(self, in_channels=2, out_channels=1):
        super(DepthRefinementUNet, self).__init__()
        
        # Encoder (채널 수 절반)
        self.encoder1 = self._conv_block(in_channels, 32)
        self.encoder2 = self._conv_block(32, 64)
        self.encoder3 = self._conv_block(64, 128)
        self.encoder4 = self._conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoder (deconvolution 후 채널 수에 맞춤)
        self.decoder4 = self._conv_block(256 + 256, 256)  # 256 + 256 = 512
        self.decoder3 = self._conv_block(128 + 128, 128)  # 128 + 128 = 256
        self.decoder2 = self._conv_block(64 + 64, 64)     # 64 + 64 = 128
        self.decoder1 = self._conv_block(32 + 32, 32)     # 32 + 32 = 64
        
        # Final output
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        
        # Upsampling with interpolate + conv
        self.upsample_b_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upsample_d4_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsample_d3_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upsample_d2_conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Radar points embedding adapters
        self.radar_pts_adapter_512 = nn.Linear(8, 512)  # for bottleneck
        self.radar_pts_adapter_128 = nn.Linear(8, 128)  # for d3
    
    def _conv_block(self, in_channels, out_channels):
        """Convolution block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upsample_block(self, x, target_size, conv_layer):
        """Upsample using interpolate + conv"""
        upsample = F.interpolate(x, size=target_size, mode='nearest')
        conv = conv_layer(upsample)
        return conv
    
    def forward(self, x, valid_radar_pts_emb):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Process radar points embedding for bottleneck (512 dim)
        # valid_radar_pts_emb: [B, 8] -> [B, 512]
        B = valid_radar_pts_emb.shape[0]
        radar_emb_adapted_512 = self.radar_pts_adapter_512(valid_radar_pts_emb)  # [B, 512]
        
        # Expand to spatial dimensions
        radar_global_512 = radar_emb_adapted_512.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 1, 1]
        radar_global_512 = radar_global_512.expand_as(b)  # [B, 512, H, W]
        
        b = b + radar_global_512
        
        # Decoder with skip connections (interpolate + conv 사용)
        up_b = self._upsample_block(b, e4.shape[2:], self.upsample_b_conv)
        d4 = self.decoder4(torch.cat([up_b, e4], dim=1))
        
        up_d4 = self._upsample_block(d4, e3.shape[2:], self.upsample_d4_conv)
        d3 = self.decoder3(torch.cat([up_d4, e3], dim=1))
        
        # Process radar points embedding for d3 (128 dim)
        radar_emb_adapted_128 = self.radar_pts_adapter_128(valid_radar_pts_emb)  # [B, 128]
        
        # Expand to spatial dimensions
        radar_global_128 = radar_emb_adapted_128.unsqueeze(-1).unsqueeze(-1)  # [B, 128, 1, 1]
        radar_global_128 = radar_global_128.expand_as(d3)  # [B, 128, H, W]
        
        d3 = d3 + radar_global_128
        
        up_d3 = self._upsample_block(d3, e2.shape[2:], self.upsample_d3_conv)
        d2 = self.decoder2(torch.cat([up_d3, e2], dim=1))
        
        up_d2 = self._upsample_block(d2, e1.shape[2:], self.upsample_d2_conv)
        d1 = self.decoder1(torch.cat([up_d2, e1], dim=1))
        
        # Final output
        output = self.final_conv(d1)
        
        # 최종 출력 크기를 900x1600으로 조정
        if output.shape[2:] != (900, 1600):
            output = F.interpolate(output, size=(900, 1600), mode='nearest')
        
        return output

class FullyConnected(torch.nn.Module):
    '''
    Fully connected layer

    Arg(s):
        in_channels : int
            number of input neurons
        out_channels : int
            number of output neurons
        dropout_rate : float
            probability to use dropout
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 dropout_rate=0.00):
        super(FullyConnected, self).__init__()

        self.fully_connected = torch.nn.Linear(in_features, out_features)

        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.fully_connected.weight)

        self.activation_func = activation_func

        if dropout_rate > 0.00 and dropout_rate <= 1.00:
            self.dropout = torch.nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        fully_connected = self.fully_connected(x)

        if self.activation_func is not None:
            fully_connected = self.activation_func(fully_connected)

        if self.dropout is not None:
            return self.dropout(fully_connected)
        else:
            return fully_connected

class FullyConnectedEncoder(torch.nn.Module):
    '''
    Fully connected encoder
    Arg(s):
        input_channels : int
            number of input channels
        n_neurons : list[int]
            number of filters to use per layer
        latent_size : int
            number of output neuron
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
    '''

    def __init__(self,
                 input_channels=3,
                 n_neurons=[32, 64, 96, 64, 32],
                 latent_size=8,
                 weight_initializer='kaiming_uniform',
                 ):
        super(FullyConnectedEncoder, self).__init__()

        activation_func = torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)

        self.mlp = torch.nn.Sequential(
            FullyConnected(
                in_features=input_channels,
                out_features=n_neurons[0],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[0],
                out_features=n_neurons[1],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[1],
                out_features=n_neurons[2],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[2],
                out_features=n_neurons[3],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[3],
                out_features=n_neurons[4],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            FullyConnected(
                in_features=n_neurons[4],
                out_features=latent_size,
                weight_initializer=weight_initializer,
                activation_func=activation_func,))

    def forward(self, x):

        return self.mlp(x)


class RadarDC(nn.Module):
    """RadarDC 모델 클래스"""
    def __init__(self):
        super(RadarDC, self).__init__()
        
        
        # Radar elevation learner (Cross-attention only)
        # self.radar_elevation_learner = RadarElevationLearner(
        #     hidden_dim=1,  # W (width) 차원에 맞춤
        #     num_heads=1   # Multi-head attention의 head 수
        # )
        
        # Point Cloud Fully Connected Layer
        self.point_cloud_fc = FullyConnectedEncoder(input_channels=3, latent_size=8)
        
        # Depth refinement UNet (refined_mde + elevation + image를 입력으로 받음)
        self.depth_refinement_unet = DepthRefinementUNet(in_channels=5, out_channels=1)  # 1 + 1 + 3 = 5
        
        # Learnable scale and shift parameters for final depth adjustment
        self.scale_factor = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.shift_factor = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        
    def create_patches_conv(self, input_tensor):
        B, C, H, W = input_tensor.shape
        patches = []
        for i in range(W):  # width 방향으로 sliding
            patch = input_tensor[:, :, :, i:i+1]  # [B, C, H, 1]
            patches.append(patch)
        return torch.stack(patches, dim=0)  # [W, B, C, H, 1]
    
    def create_points_from_radar(self, radar_tensor, max_points=None):
        """
        elevation_learned_radar를 포인트클라우드 형태로 변환
        
        Args:
            radar_tensor: [B, 1, H, W] - elevation learned radar tensor
            max_points: 최대 포인트 수 (패딩용, None이면 동적 크기)
            
        Returns:
            points_tensor: [B, K, 3] where K is max number of points across all batches
        """
        points_list = []
        B, _, H, W = radar_tensor.shape
        
        for b in range(B):
            radar_map = radar_tensor[b, 0]  # [H, W]
            
            # 유효한 pixel mask (0이 아닌 depth 추출)
            valid_mask = (radar_map != 0)
            
            if valid_mask.sum() > 0:
                # (y_idx, x_idx) 좌표
                y_idx, x_idx = torch.where(valid_mask)
                z = radar_map[y_idx, x_idx]  # 해당 pixel의 depth값
                
                # [N, 3] 형태로 point cloud 구성 (x, y, z)
                xyz = torch.stack([x_idx.float(), y_idx.float(), z.float()], dim=1)  # [N, 3]
            else:
                # Empty point cloud
                xyz = torch.empty((0, 3), device=radar_map.device)
            
            points_list.append(xyz)
        
        # 최대 포인트 수 계산
        if max_points is None:
            max_points = max([p.shape[0] for p in points_list]) if points_list else 0
        
        # 패딩하여 [B, K, 3] 형태로 변환
        padded_points = []
        for points in points_list:
            if points.shape[0] == 0:
                # 빈 포인트클라우드인 경우 0으로 패딩
                padded = torch.zeros((max_points, 3), device=points.device)
            else:
                if points.shape[0] >= max_points:
                    # 잘라내기
                    padded = points[:max_points]
                else:
                    # 패딩
                    padding = torch.zeros((max_points - points.shape[0], 3), device=points.device)
                    padded = torch.cat([points, padding], dim=0)
            padded_points.append(padded)
        
        return torch.stack(padded_points, dim=0)  # [B, K, 3]
    
    def create_mde_features_for_fusion(self, mde_features, target_shape):
        """
        mde_features를 attention fusion용 포인트클라우드 형태로 변환
        
        Args:
            mde_features: [B, 384, 42, 74] - MDE features
            target_shape: (B, C, H, W) - target shape for interpolation
            
        Returns:
            feature_points_mde: List of processed MDE features, each with shape [N, 256]
        """
        B, C, H, W = target_shape
        feature_points_mde = []
        
        # mde_features를 target shape로 interpolate
        mde_features_resized = F.interpolate(mde_features, size=(H, W), mode='bilinear', align_corners=False)
        
        # 384 -> 256 채널로 변환
        if mde_features_resized.shape[1] != 256:
            if not hasattr(self, 'mde_channel_adapter'):
                self.mde_channel_adapter = nn.Conv2d(384, 256, kernel_size=1).to(
                    device=mde_features.device,
                    dtype=torch.float32
                )
            mde_features_resized = self.mde_channel_adapter(mde_features_resized)
        
        # 각 배치별로 spatial sampling하여 포인트클라우드 형태로 변환
        for b in range(B):
            # Spatial sampling: H*W 개의 points에서 features 추출
            img_feat = mde_features_resized[b]  # [256, H, W]
            img_feat_flat = img_feat.view(256, -1).permute(1, 0)  # [H*W, 256]
            feature_points_mde.append(img_feat_flat)
        
        return feature_points_mde
    
    def apply_attention_fusion(self, radar_features, image_features):
        """
        Radar와 image features 간의 attention fusion (RCNet 방식 참고)
        
        Args:
            radar_features: List of radar features [N, 256]
            image_features: List of image features [H*W, 256]
            
        Returns:
            fused_features: List of fused features
        """
        fused_features = []
        
        for b in range(len(radar_features)):
            radar_feat = radar_features[b]  # [N, 256]
            image_feat = image_features[b]  # [H*W, 256]
            
            if radar_feat.numel() > 0 and image_feat.numel() > 0:
                # RCNet 방식: attention 적용 후 concatenation
                # Reshape for attention: [N, 256] -> [1, N, 256], [H*W, 256] -> [1, H*W, 256]
                radar_feat_batch = radar_feat.unsqueeze(0)  # [1, N, 256]
                image_feat_batch = image_feat.unsqueeze(0)  # [1, H*W, 256]
                
                # Simple attention fusion (can be replaced with more sophisticated attention)
                # For now, use identity mapping (no attention)
                radar_feat_attended = radar_feat_batch
                image_feat_attended = image_feat_batch
                
                # Concatenate attended features: [1, N, 256] + [1, H*W, 256] -> [1, N+H*W, 256]
                fused_feat = torch.cat([radar_feat_attended, image_feat_attended], dim=1)  # [1, N+H*W, 256]
                fused_feat = fused_feat.squeeze(0)  # [N+H*W, 256]
            elif radar_feat.numel() > 0:
                fused_feat = radar_feat
            elif image_feat.numel() > 0:
                fused_feat = image_feat
            else:
                fused_feat = torch.empty((0, 256), device=radar_feat.device)
            
            fused_features.append(fused_feat)
        
        return fused_features
    
    def convert_fused_features_to_spatial(self, fused_features, target_shape):
        """
        Fused features를 spatial depth map으로 변환
        
        Args:
            fused_features: List of fused features [N+H*W, 256]
            target_shape: (B, C, H, W) - target spatial shape
            
        Returns:
            fused_depth: [B, 1, H, W] - spatial depth map
        """
        B, C, H, W = target_shape
        fused_depth_maps = []
        
        for b in range(B):
            fused_feat = fused_features[b]  # [N+H*W, 256]
            
            if fused_feat.numel() > 0:
                # Global pooling으로 1차원 feature 생성
                global_feat = fused_feat.mean(dim=0)  # [256]
                # Spatial 형태로 확장: [256] -> [256, H, W]
                spatial_feat = global_feat.view(256, 1, 1).expand(256, H, W)  # [256, H, W]
            else:
                # Empty case
                spatial_feat = torch.zeros(256, H, W, device=fused_feat.device)
            
            fused_depth_maps.append(spatial_feat)
        
        # Batch로 합치기
        fused_depth_maps = torch.stack(fused_depth_maps, dim=0)  # [B, 256, H, W]
        
        # Fused features를 1채널로 변환
        if not hasattr(self, 'fused_to_depth_adapter'):
            self.fused_to_depth_adapter = nn.Conv2d(256, 1, kernel_size=1).to(
                device=fused_depth_maps.device,
                dtype=torch.float32
            )
        
        fused_depth = self.fused_to_depth_adapter(fused_depth_maps)  # [B, 1, H, W]
        
        return fused_depth
    
    def apply_distance_based_refinement(self, mde_output, elevation_learned_radar, elevation_mask):
        """
        Distance-based refinement: elevation 주변의 MDE 값을 elevation 값으로 influence
        
        Args:
            mde_output: [B, 1, H, W] - MDE depth output
            elevation_learned_radar: [B, 1, H, W] - elevation learned radar
            elevation_mask: [B, 1, H, W] - elevation valid mask
            
        Returns:
            refined_mde: [B, 1, H, W] - refined MDE output
        """
        B, C, H, W = mde_output.shape
        refined_mde = mde_output.clone()
        
        # Gaussian kernel for spatial influence
        kernel_size = 5
        sigma = 1.0
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(kernel_size, device=mde_output.device, dtype=torch.float32),
            torch.arange(kernel_size, device=mde_output.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Calculate distances from center
        center = kernel_size // 2
        distances_squared = (y_coords - center) ** 2 + (x_coords - center) ** 2
        
        # Create Gaussian kernel
        gaussian_kernel = torch.exp(-distances_squared / (2 * sigma ** 2))
        
        # Normalize kernel
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        for b in range(B):
            # Get elevation values and their positions
            elevation_values = elevation_learned_radar[b, 0]  # [H, W]
            elevation_positions = (elevation_values > 0).float()  # [H, W]
            
            if elevation_positions.sum() == 0:
                continue  # No elevation data for this batch
                
            # Apply spatial influence: elevation 주변의 MDE 값을 elevation 값으로 influence
            # 1. elevation이 있는 곳은 직접 elevation 값으로 교체
            refined_mde[b, 0] = torch.where(
                elevation_positions.bool(),
                elevation_values,
                refined_mde[b, 0]
            )
            
            # 2. elevation 주변의 MDE 값을 elevation 값에 가깝게 조정 (soft influence)
            # Convolution을 사용하여 elevation의 spatial influence 적용
            elevation_influence = F.conv2d(
                elevation_learned_radar[b:b+1], 
                gaussian_kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size//2
            )[0, 0]  # [H, W]
            
            # MDE 값을 elevation influence에 따라 조정
            # influence가 강한 곳일수록 elevation 값에 가깝게 조정
            influence_weight = torch.sigmoid(elevation_influence * 2.0)  # [H, W]
            refined_mde[b, 0] = (1 - influence_weight) * mde_output[b, 0] + influence_weight * elevation_influence
        
        return refined_mde
    
    def apply_attention_based_refinement(self, mde_output, elevation_learned_radar, elevation_mask):
        """
        Attention-based refinement: elevation과 MDE 간의 attention을 사용하여 refinement
        
        Args:
            mde_output: [B, 1, H, W] - MDE depth output
            elevation_learned_radar: [B, 1, H, W] - elevation learned radar
            elevation_mask: [B, 1, H, W] - elevation valid mask
            
        Returns:
            refined_mde: [B, 1, H, W] - attention-refined MDE output
        """
        B, C, H, W = mde_output.shape
        
        # Flatten spatial dimensions for attention
        mde_flat = mde_output.view(B, C, -1)  # [B, 1, H*W]
        elevation_flat = elevation_learned_radar.view(B, C, -1)  # [B, 1, H*W]
        mask_flat = elevation_mask.view(B, C, -1)  # [B, 1, H*W]
        
        # Compute attention weights between MDE and elevation
        # Similarity-based attention
        similarity = F.cosine_similarity(mde_flat, elevation_flat, dim=1)  # [B, H*W]
        attention_weights = F.softmax(similarity, dim=1)  # [B, H*W]
        
        # Apply attention: elevation이 valid한 곳은 더 높은 attention
        attention_weights = attention_weights * mask_flat.squeeze(1)  # [B, H*W]
        attention_weights = attention_weights.unsqueeze(1)  # [B, 1, H*W]
        
        # Refined MDE: attention-weighted combination
        refined_flat = attention_weights * elevation_flat + (1 - attention_weights) * mde_flat
        
        # Reshape back to spatial
        refined_mde = refined_flat.view(B, C, H, W)
        
        return refined_mde
    
    def forward(self, mde_output, mde_features, radar, image):
        # Convert mde_features from Half precision to float32 to avoid dtype mismatch
        # mde_features = mde_features.float()
        mde_output = mde_output * self.scale_factor + self.shift_factor


        radar_mask = (radar != 0)
        valid_count = radar_mask.sum()  # 전체 valid 포인트 개수

        
        # Radar elevation learning을 위한 patch 생성
        radar_patches = self.create_patches_conv(radar)  # [W, B, C, H, 1]
        mde_out_patches = self.create_patches_conv(mde_output)  # [W, B, C, H, 1]
        
        # Radar elevation learning 적용
        # elevation_learned_radar = self.radar_elevation_learner(radar_patches, mde_out_patches)  # [B, 1, H, W]

        # Compute similarity between radar and vp_ground_truth (same position comparison)
        similarity_scores = compute_similarity(radar_patches, mde_out_patches)         
        # Create radar_GT based on similarity
        elevation_learned_radar = create_radar_gt(radar_patches, mde_out_patches, similarity_scores)
        

        elevation_learned_radar_mask = (elevation_learned_radar != 0)
        valid_count_elevation_learned_radar = elevation_learned_radar_mask.sum()  # 전체 valid 포인트 개수

        num_point_invariant = True if valid_count == valid_count_elevation_learned_radar else False
        

        # elevation_learned_radar를 포인트클라우드 형태로 변환
        points_elevation_learned_radar = self.create_points_from_radar(elevation_learned_radar)
        B,K,C = points_elevation_learned_radar.shape
        points_elevation_learned_radar = points_elevation_learned_radar.reshape(B*K, -1)
        radar_pts_emb = self.point_cloud_fc(points_elevation_learned_radar)
        radar_pts_emb = radar_pts_emb.reshape(B,K,radar_pts_emb.shape[-1])
        valid_radar_pts_cnts = radar_pts_emb.shape[1]

        valid_radar_pts_emb = []
        for _radar_pts_emb in radar_pts_emb:
            valid_radar_pts_emb.append(_radar_pts_emb[:valid_radar_pts_cnts,:].mean(axis=0))
        valid_radar_pts_emb = torch.stack(valid_radar_pts_emb, dim=0)  # [B, 8]



        # # mde_features를 attention fusion용으로 가공
        # feature_points_mde = self.create_mde_features_for_fusion(mde_features, elevation_learned_radar.shape)

        # # Attention fusion between radar and image features
        # fused_features = self.apply_attention_fusion(feature_points_elevation_learned_radar, feature_points_mde)
        # fused_depth = self.convert_fused_features_to_spatial(fused_features, elevation_learned_radar.shape)

        # Elevation-guided refinement: elevation_learned_radar의 valid value를 기준으로 MDE refinement
        # 1. Confidence mask 생성 (elevation_learned_radar가 valid한 곳은 높은 confidence)
        elevation_mask = (elevation_learned_radar > 0).float()  # [B, 1, H, W]
        
        # 2. Refinement 방법 선택 (Distance-based 또는 Attention-based)
        # Distance-based refinement: elevation 주변의 MDE 값을 elevation 값으로 influence
        refined_mde = self.apply_distance_based_refinement(mde_output, elevation_learned_radar, elevation_mask)
        
        # Alternative: Attention-based refinement (주석 해제하여 사용 가능)
        # refined_mde = self.apply_attention_based_refinement(mde_output, elevation_learned_radar, elevation_mask)
        
        # 3. Confidence-weighted blending
        # elevation이 있는 곳은 더 높은 weight를 가짐
        confidence_weight = elevation_mask * 0.8 + (1 - elevation_mask) * 0.2  # [B, 1, H, W]
        final_refined_mde = confidence_weight * refined_mde + (1 - confidence_weight) * mde_output
        
        # 4. UNet 입력 구성: refined MDE + elevation + image
        unet_input = torch.cat([final_refined_mde, elevation_learned_radar, image], dim=1)  # [B, 5, H, W]
        final_depth = self.depth_refinement_unet(unet_input, valid_radar_pts_emb)

        # Apply learnable scale and shift to final depth
        
        

        return final_depth, elevation_learned_radar, num_point_invariant
    
