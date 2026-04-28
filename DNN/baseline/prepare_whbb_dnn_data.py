import time
import numpy as np
import uproot


def prepare_whbb_dnn_data(
    filename="/home/haoran/inputs.root",
    treename="Nominal",
    output_npz="whbb_dnn_prepared.npz",
    step_size="512 MB",
    save_output=True,
):
    """
    读取 WHbb ntuple，做最基本的预处理和统计检查。

    功能：
      - 分块读取 ROOT TTree
      - 做 cut: nTags == 2, nJ == 2
      - 构造二分类标签：
            sample == "qqWlvH125" -> signal (1)
            else                  -> background (0)
      - 保存训练真正需要的变量
      - 打印统计信息
      - 可选输出为 npz 文件，供后续 PyTorch 训练使用
    """

    t0 = time.time()

    # ------------------------------------------------------------
    # 1. 定义要读取的 branch
    # ------------------------------------------------------------
    feature_names = [
        "mBB",
        "dRBB",
        "pTV",
        "pTB1",
        "pTB2",
        "bin_MV2c10B1",
        "bin_MV2c10B2",
        "dPhiVBB",
        "MET",
    ]

    extra_names = [
        "sample",
        "EventWeight",
        "EventNumber",
        "nTags",
        "nJ",
    ]

    branches = feature_names + extra_names

    print("=" * 80)
    print("WHbb DNN data preparation")
    print(f"Input file : {filename}")
    print(f"Tree name  : {treename}")
    print(f"Step size  : {step_size}")
    print("=" * 80)

    # ------------------------------------------------------------
    # 2. 用 list 收集每个 chunk 通过 cut 后的数据
    # ------------------------------------------------------------
    x_chunks = []
    y_chunks = []
    w_chunks = []
    evn_chunks = []
    sample_chunks = []

    # 统计量
    total_events_read = 0
    total_events_after_cut = 0

    total_sig_count = 0
    total_bkg_count = 0

    total_sig_weight = 0.0
    total_bkg_weight = 0.0

    odd_count = 0
    even_count = 0

    # 每个变量的全局 min/max
    feature_min = {name: np.inf for name in feature_names}
    feature_max = {name: -np.inf for name in feature_names}

    # ------------------------------------------------------------
    # 3. 分块读取
    # ------------------------------------------------------------
    tree_path = f"{filename}:{treename}"

    chunk_id = 0
    for arrays in uproot.iterate(
        tree_path,
        branches,
        library="np",
        step_size=step_size,
    ):
        chunk_id += 1

        # 当前 chunk 的原始事件数
        n_chunk = len(arrays["EventNumber"])
        total_events_read += n_chunk

        print(f"[Chunk {chunk_id}] raw events = {n_chunk:,}")

        # --------------------------------------------------------
        # 4. 做 cut
        # --------------------------------------------------------
        mask = (
            (arrays["nTags"] == 2)
            & (arrays["nJ"] == 2)
        )

        n_after = int(np.sum(mask))
        total_events_after_cut += n_after

        print(f"[Chunk {chunk_id}] after cuts = {n_after:,}")

        # 如果这一块一个事件都没留下，直接跳过
        if n_after == 0:
            continue

        # --------------------------------------------------------
        # 5. 取出 cut 后的变量
        # --------------------------------------------------------
        # 10 个输入变量拼成一个二维数组 X，形状为 [N, 10]
        x_chunk = np.column_stack([arrays[name][mask] for name in feature_names]).astype(np.float32)

        # sample 处理：uproot 读出来可能是 bytes，也可能是 str
        # 我们统一转成 Python 字符串数组
        raw_sample = arrays["sample"][mask]
        sample_chunk = np.array([
            s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
            for s in raw_sample
        ])

        # 构造 label：qqWlvH125 -> 1, else 0
        y_chunk = (sample_chunk == "qqWlvH125").astype(np.int64)

        # 权重和事件号
        w_chunk = arrays["EventWeight"][mask].astype(np.float32)
        evn_chunk = arrays["EventNumber"][mask].astype(np.int64)

        # --------------------------------------------------------
        # 6. 更新统计信息
        # --------------------------------------------------------
        n_sig = int(np.sum(y_chunk == 1))
        n_bkg = int(np.sum(y_chunk == 0))

        total_sig_count += n_sig
        total_bkg_count += n_bkg

        if n_sig > 0:
            total_sig_weight += float(np.sum(w_chunk[y_chunk == 1]))
        if n_bkg > 0:
            total_bkg_weight += float(np.sum(w_chunk[y_chunk == 0]))

        odd_mask = (evn_chunk % 2 == 1)
        even_mask = (evn_chunk % 2 == 0)

        odd_count += int(np.sum(odd_mask))
        even_count += int(np.sum(even_mask))

        # 更新 feature 范围
        for i, name in enumerate(feature_names):
            this_min = float(np.min(x_chunk[:, i]))
            this_max = float(np.max(x_chunk[:, i]))

            if this_min < feature_min[name]:
                feature_min[name] = this_min
            if this_max > feature_max[name]:
                feature_max[name] = this_max

        # --------------------------------------------------------
        # 7. 保存这一块的数据
        # --------------------------------------------------------
        x_chunks.append(x_chunk)
        y_chunks.append(y_chunk)
        w_chunks.append(w_chunk)
        evn_chunks.append(evn_chunk)
        sample_chunks.append(sample_chunk)

    # ------------------------------------------------------------
    # 8. 拼接所有 chunk
    # ------------------------------------------------------------
    print("\nConcatenating all selected chunks ...")

    if len(x_chunks) == 0:
        print("ERROR: No events survived the cuts.")
        return None

    X = np.concatenate(x_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    w = np.concatenate(w_chunks, axis=0)
    event_number = np.concatenate(evn_chunks, axis=0)
    sample = np.concatenate(sample_chunks, axis=0)

    # ------------------------------------------------------------
    # 9. 打印总统计
    # ------------------------------------------------------------
    t1 = time.time()

    print("\n" + "=" * 80)
    print("Final summary")
    print("=" * 80)
    print(f"Total raw events read      : {total_events_read:,}")
    print(f"Total events after cuts    : {total_events_after_cut:,}")
    print(f"Final concatenated events  : {len(X):,}")
    print()

    print(f"Signal events (count)      : {total_sig_count:,}")
    print(f"Background events (count)  : {total_bkg_count:,}")
    print()

    print(f"Signal weight sum          : {total_sig_weight:.6e}")
    print(f"Background weight sum      : {total_bkg_weight:.6e}")
    print()

    print(f"Odd events                 : {odd_count:,}")
    print(f"Even events                : {even_count:,}")
    print()

    print("Feature ranges after cuts:")
    for name in feature_names:
        print(f"  {name:15s}: [{feature_min[name]:12.6f}, {feature_max[name]:12.6f}]")

    print()
    print(f"Elapsed time               : {t1 - t0:.2f} s")
    print("=" * 80)

    # ------------------------------------------------------------
    # 10. 可选保存输出
    # ------------------------------------------------------------
    if save_output:
        print(f"\nSaving prepared arrays to: {output_npz}")

        np.savez_compressed(
            output_npz,
            X=X,
            y=y,
            w=w,
            event_number=event_number,
            sample=sample,
            feature_names=np.array(feature_names, dtype=object),
        )

        print("Saved successfully.")

    # ------------------------------------------------------------
    # 11. 返回字典，方便交互式调试
    # ------------------------------------------------------------
    return {
        "X": X,
        "y": y,
        "w": w,
        "event_number": event_number,
        "sample": sample,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    data = prepare_whbb_dnn_data(
        filename="/home/haoran/inputs.root",
        treename="Nominal",
        output_npz="whbb_dnn_prepared.npz",
        step_size="512 MB",
        save_output=True,
    )
