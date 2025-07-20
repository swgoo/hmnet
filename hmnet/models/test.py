import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch.nn.functional as F

def calculate_chunk_boundaries_from_mask(boundary_mask, device=None):
    """
    boundary_mask에서 청크 경계를 계산
    boundary_mask: [seq_len] 텐서, 1이면 경계 (직전 0들과 같은 청크)
    """
    if device is None:
        device = boundary_mask.device if hasattr(boundary_mask, 'device') else torch.device('cpu')
    
    boundary_mask = torch.tensor(boundary_mask, device=device, dtype=torch.bool)
    seq_len = len(boundary_mask)
    
    # 청크 경계 위치 찾기 (1인 위치들)
    boundary_positions = torch.where(boundary_mask)[0]
    
    # 청크 시작 위치들 계산
    chunk_starts = [0]  # 첫 번째 청크는 항상 0에서 시작
    if len(boundary_positions) > 0:
        # 경계 다음 위치들이 새로운 청크의 시작
        chunk_starts.extend((boundary_positions + 1).tolist())
    
    # 청크 끝 위치들 계산
    chunk_ends = boundary_positions.tolist() + [seq_len - 1]
    
    # 청크 길이들 계산
    chunk_lengths = []
    for i in range(len(chunk_starts)):
        if i < len(chunk_ends):
            chunk_lengths.append(chunk_ends[i] - chunk_starts[i] + 1)
    
    # 청크 경계들 (cumsum)
    chunk_boundaries = [0] + [sum(chunk_lengths[:i+1]) for i in range(len(chunk_lengths))]
    
    return torch.tensor(chunk_boundaries, device=device, dtype=torch.long), chunk_lengths

def create_chunked_causal_block_mask(
    seq_len: int,
    boundary_mask: list,
    block_score: torch.Tensor,
    threshold: float = 0.5,
    device: torch.device = None
):
    """
    청킹된 시퀀스를 위한 causal block mask 생성 (확률 기반)
    boundary_mask: [seq_len] 리스트/텐서, 1이면 경계 (직전 0들과 같은 청크)
    block_score: [num_chunks, num_chunks] 확률 텐서
    threshold: 블록 허용 임계값 (기본 0.5)
    """
    if device is None:
        device = torch.device('cpu')
    
    # boundary_mask에서 청크 경계 계산
    chunk_boundaries, chunk_lengths = calculate_chunk_boundaries_from_mask(boundary_mask, device)
    
    # block_score를 텐서로 변환하고 임계값으로 이진화
    block_selection_tensor = (block_score > threshold).to(device)
    
    num_chunks = len(chunk_lengths)
    
    def causal_chunked_mask(b, h, q_idx, kv_idx):
        """
        텐서 연산으로만 구성된 마스크 함수 (조건문 없음)
        """
        # causal mask: kv_idx <= q_idx 조건
        causal_condition = kv_idx <= q_idx
        
        # 각 위치가 어느 청크에 속하는지 계산
        q_expanded = q_idx.expand(num_chunks)
        kv_expanded = kv_idx.expand(num_chunks)
        
        # 청크 경계 비교
        q_ge_start = q_expanded >= chunk_boundaries[:-1]  # [num_chunks]
        q_lt_end = q_expanded < chunk_boundaries[1:]      # [num_chunks]
        q_in_chunk = q_ge_start & q_lt_end                # [num_chunks]
        
        kv_ge_start = kv_expanded >= chunk_boundaries[:-1]  # [num_chunks]
        kv_lt_end = kv_expanded < chunk_boundaries[1:]      # [num_chunks]
        kv_in_chunk = kv_ge_start & kv_lt_end               # [num_chunks]
        
        # 청크 조합별 허용 여부 계산 (확률 > threshold)
        chunk_pairs = q_in_chunk.unsqueeze(1) & kv_in_chunk.unsqueeze(0)  # [num_chunks, num_chunks]
        allowed_pairs = chunk_pairs & block_selection_tensor              # [num_chunks, num_chunks]
        chunk_allowed = torch.any(allowed_pairs)
        
        # 최종 마스크: causal 조건 AND 청크 허용 조건
        return causal_condition & chunk_allowed
    
    # create_block_mask 사용
    block_mask = create_block_mask(
        causal_chunked_mask,
        B=None,
        H=None,  
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device
    )
    
    return block_mask

def create_chunked_score_mod(
    boundary_mask: list,
    block_score: torch.Tensor,
    device: torch.device = None
):
    """
    청킹된 시퀀스를 위한 score_mod 함수 생성
    boundary_mask: [seq_len] 리스트/텐서, 1이면 경계 (직전 0들과 같은 청크)
    block_score: [num_chunks, num_chunks] 확률 텐서
    """
    if device is None:
        device = torch.device('cpu')
    
    # boundary_mask에서 청크 경계 계산
    chunk_boundaries, chunk_lengths = calculate_chunk_boundaries_from_mask(boundary_mask, device)
    
    # block_score를 텐서로 변환
    block_score_tensor = block_score.to(device)
    
    num_chunks = len(chunk_lengths)
    
    def chunked_score_mod(score, b, h, q_idx, kv_idx):
        """
        청크별 확률을 적용하는 score_mod 함수
        """
        # 각 위치가 어느 청크에 속하는지 계산
        q_expanded = q_idx.expand(num_chunks)
        kv_expanded = kv_idx.expand(num_chunks)
        
        # 청크 경계 비교
        q_ge_start = q_expanded >= chunk_boundaries[:-1]  # [num_chunks]
        q_lt_end = q_expanded < chunk_boundaries[1:]      # [num_chunks]
        q_in_chunk = q_ge_start & q_lt_end                # [num_chunks]
        
        kv_ge_start = kv_expanded >= chunk_boundaries[:-1]  # [num_chunks]
        kv_lt_end = kv_expanded < chunk_boundaries[1:]      # [num_chunks]
        kv_in_chunk = kv_ge_start & kv_lt_end               # [num_chunks]
        
        # 해당하는 청크 조합의 확률값 찾기
        q_chunk_idx = torch.argmax(q_in_chunk.float())
        kv_chunk_idx = torch.argmax(kv_in_chunk.float())
        
        # 해당 청크 조합의 확률값 가져오기
        prob_score = block_score_tensor[q_chunk_idx, kv_chunk_idx]
        
        # 원래 점수에 확률을 곱해서 반환
        return score * prob_score
    
    return chunked_score_mod

def chunked_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    boundary_mask: list,
    block_score: torch.Tensor,
    threshold: float = 0.5,
    use_score_mod: bool = True,
    scale: float = None
):
    """
    청킹된 FlexAttention 실행 (확률 기반 block_mask와 score_mod)
    boundary_mask: [seq_len] 리스트/텐서, 1이면 경계 (직전 0들과 같은 청크)
    block_score: [num_chunks, num_chunks] 확률 텐서
    threshold: 블록 허용 임계값
    use_score_mod: score_mod 사용 여부
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # 확률 기반 block mask 생성
    block_mask = create_chunked_causal_block_mask(
        seq_len=seq_len,
        boundary_mask=boundary_mask,
        block_score=block_score,
        threshold=threshold,
        device=query.device
    )
    
    # score_mod 함수 생성
    score_mod = None
    if use_score_mod:
        score_mod = create_chunked_score_mod(
            boundary_mask=boundary_mask,
            block_score=block_score,
            device=query.device
        )
    
    # FlexAttention 실행
    output = flex_attention(
        query, key, value,
        block_mask=block_mask,
        score_mod=score_mod,
        scale=scale
    )
    
    return output

# 테스트 코드
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 1
    num_heads = 2
    seq_len = 8
    head_dim = 4

    # 임의의 쿼리, 키, 밸류 생성
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # boundary_mask: 1이면 경계, 0이면 내부
    # 예: [0, 0, 1, 0, 1, 0, 0, 1] → 청크: [0,1,2][3,4][5,6,7]
    boundary_mask = [0, 0, 1, 0, 1, 0, 0, 1]

    # 청크 개수 계산
    chunk_boundaries, chunk_lengths = calculate_chunk_boundaries_from_mask(boundary_mask)
    num_chunks = len(chunk_lengths)

    # 임의의 block_score 생성 (num_chunks x num_chunks)
    block_score = torch.rand(num_chunks, num_chunks)

    # chunked_flex_attention 실행
    output = chunked_flex_attention(
        query=query,
        key=key,
        value=value,
        boundary_mask=boundary_mask,
        block_score=block_score,
        threshold=0.5,
        use_score_mod=True
    )

    print("Output shape:", output.shape)
    print("Output:", output)
