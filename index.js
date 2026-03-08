import tf from '@tensorflow/tfjs-node';


async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    model.add(tf.layers.dense({ inputShape: [9], units: 100, activation: 'relu' }))

    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Trainning the model
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, log) => console.log(
                    `Epoch: ${epoch}: loss = ${log.loss}`
                )
            }
        }
    )

    return model
}

async function predict(model, level) {
    const tfInput = tf.tensor2d(level)

    // Start the prediction
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}
// Example of levels for training (each level with its characteristics):
// const level = [
//     { level: "beginner", chords: "basic chords", scales: "pentatonic", licks: "in just 1 shape" },
//     { level: "intermediate", chords: "CAGED", scales: "minor natural scale", licks: "in various shapes" },
//     { level: "advanced", chords: "inverted chords", scales: "minor harmonic scale", licks: "in various tones" }
// ];

// Input vectors with values already normalized and one-hot encoded
// Order: [basic chords, CAGED, inverted chords, 
//          scale pentatonic, minor natural scale, minor harmonic scale, 
//          in just 1 shape, in various shapes, in various tones]
// const tensorLevels = [
//     [1, 0, 0, 1, 0, 0, 1, 0, 0], // beginner
//     [1, 1, 0, 1, 1, 0, 1, 1, 0],    // intermediate
//     [1, 1, 1, 1, 1, 1, 1, 1, 1]     // advanced
// ]

// tensorLevels corresponds to the input dataset of the model.
const tensorLevels = [
    [1, 0, 0, 1, 0, 0, 1, 0, 0],    // beginner
    [1, 1, 0, 1, 1, 0, 1, 1, 0],    // intermediate
    [1, 1, 1, 1, 1, 1, 1, 1, 1]     // advanced
]

// Category labels to be predicted (one-hot encoded)
// [beginner, intermediate, advanced]
const guitarLevels = ["beginner", "intermediate", "advanced"];
const tensorLabels = [
    [1, 0, 0], // beginner
    [0, 1, 0], // intermediate
    [0, 0, 1]  // advanced
];

// input (xs) and output (ys) tensors to train the model
const inputXs = tf.tensor2d(tensorLevels)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs)

const guitarist = { acorde: 'basic chords', scales: 'pentatonic', licks: 'in just 1 shape' }

const levelTensorNormalized = [
    [
        1,    // basic chords
        0,    // CAGED
        0,    // inverted chords
        1,    // pentatonic scale
        0,    // natural minor scale
        0,    // harmonic minor scale
        1,    // in just 1 shape
        0,    // in various shapes
        0,    // in various tones        
    ]
]

const predictions = await predict(model, levelTensorNormalized)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${guitarLevels[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')

console.log(results)
