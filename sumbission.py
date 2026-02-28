# ==========================================
# 6. MAKE PREDICTIONS ON TEST DATA
# ==========================================

test_dataset = ImageCSVDataset(
    csv_file=os.path.join(csv_drive_path, "test.csv"), 
    root_dir=test_path,    
    transform=transform,
    is_test=True
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Inverse map to turn integers (0, 1) back into strings ("bad", "good")
idx2label = {v: k for k, v in label_map.items()}

model.eval()
preds = []
ids = []

with torch.no_grad():
    for image, image_id in test_loader:
        outputs = model(image.to(device))
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())
        ids.extend([id.item() if isinstance(id, torch.Tensor) else id for id in image_id])

pred_labels = [idx2label[p.item()] for p in preds]

submission = pd.DataFrame({
    "ID": ids,
    "target": pred_labels
})

submission.to_csv("submission.csv", index=False)
print(f"submission.csv successfully created! Ready for upload.")
