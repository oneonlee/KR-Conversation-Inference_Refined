ensemble_recipe=$1

echo "Start Hard voting Ensemble"

python -m run.test \
    --ensemble \
    --ensemble_file_list ${!ensemble_recipe} \
    --output resource/results/predictions/final-result.json

echo "End Hard voting Ensemble"