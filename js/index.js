const trainingData = [
    "leonhard mondsturm",
    "sarah tausendstern",
    "otto plapper",
    "ludwig hammerstein",
    "hektor cabana",
    "flavio faltdach",
    "monty münzfach",
    "bruno plapper",
    "jakob moorstelz",
    "korrin küstenseher",
    "wilhelm sturmwind",
    "helena sternenfels",
    "gustav nebelkrone",
    "mathilda regenbogen",
    "ethengub aririm",
    "konrad wolkenschiff",
    "siegfried bergmann",
    "emma silbertau",
    "klaus donnerhall",
    "frieda morgenstern",
    "robert windläufer",
    "gregor felsenstein",
    "hilda waldecho",
    "viktor sturmsänger",
    "berta mondlicht",
    "heinrich seefahrer",
    "else winterwind",
    "karl steinherz",
    "martha regentanz",
    "theodor blitzfänger",
    "anna sternenstaub",
    "felix wellentanz",
    "clara nebelfänger",
    "walter bergsturm",
    "lotte himmelspfad",
    "siegfried stanzhüter",
    "kerrim rammer",
    "abkort ronstein",
    "ikeb gartentor",
    "richard moorgrim",
    "ibana wehchor",
    "eod antyber",
    "olin wiegendau",
    "yboo marlitz",
    "antenhard weinherz",
    "eugena faube",
    "kena vierwelt",
    "enzian seelapper",
    "orfind ronstein",
    "sebastian silhelm",
    "innse inntanz",
    "kelkena immerwind",
    "frita megenturm",
    "anhantus monnerhalle",
    "rier züsenstein",
    "ena megentanz",
    "ronis hicht",
    "Torben Krautfass",
    "Viktor Schlenker",
    "Helgen Kettlemund",
    "Frobin Feldsprang",
    "Milda Wurzleim",
    "Ornib Klippenschwing",
    "Ethelbert Planzruch",
    "Jorgen Moosbrant",
    "Ida Watterspitz",
    "Lomren Stüpfelzug",
    "Evarin Glanzbüttel",
    "Gerda Tiefgrund",
    "Branthos Keilenstab",
    "Elthar Viltorn",
    "Hildor Bachschnurr",
    "Syrith Wogenhain",
    "Roderick Plundelwacht",
    "Yennith Kurzwand",
    "Finnick Krähenbaum",
    "Lorta Moonscheid",
    "Grondar Sägebrot",
    "Ansel Wartenthor",
]

// Model variables
let model, characterToIndex, indexToCharacter, maxLength, vocabularySize
let totalEpochsTrained = 0
const AMOUNT_EPOCHS = 100

// Adjust UI to variables
document.querySelector('#trainingData').value = trainingData.join('\n')
document.querySelector('#train').innerText = `Train for ${AMOUNT_EPOCHS} epochs`

// Preprocess text data
const preprocessData = (textArray) => {
    // Get all unique characters
    const uniqueChars = Array.from(new Set(textArray.join('').toLowerCase()))
    
    // Create character to index and index to character mappings
    const characterToIndex = {}
    const indexToCharacter = {}
    uniqueChars.forEach((character, index) => {
        characterToIndex[character] = index
        indexToCharacter[index] = character
    })
    
    // Convert text to sequences
    const sequences = []
    const nextChars = []

    textArray.forEach(text => {
        const lowerText = text.toLowerCase()
        for (let i = 0; i < lowerText.length - 1; i++) {
            sequences.push(lowerText.slice(0, i + 1))
            nextChars.push(lowerText[i + 1])
        }
    })
    
    const maxLength = Math.max(...sequences.map(seq => seq.length))
    
    // Prepare input data
    const X = sequences.map(seq => {
        const indices = Array(maxLength).fill(0);
        [...seq].forEach((char, i) => {
            indices[i] = characterToIndex[char]
        })
        return indices
    })
    
    // Prepare output data
    const Y = nextChars.map(char => {
        const output = Array(uniqueChars.length).fill(0)
        output[characterToIndex[char]] = 1
        return output
    })
    
    return {
        X: tf.tensor2d(X, [X.length, maxLength]),
        y: tf.tensor2d(Y, [Y.length, uniqueChars.length]),
        characterToIndex: characterToIndex,
        indexToCharacter: indexToCharacter,
        maxLength: maxLength,
        vocabularySize: uniqueChars.length
    }
}

// Create LSTM model
const createModel = (maxLen, vocabSize) => {
    const model = tf.sequential()
    
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: 32,
        inputLength: maxLen
    }))
    
    model.add(tf.layers.lstm({
        units: 128,
        returnSequences: true
    }))
    
    model.add(tf.layers.lstm({
        units: 128
    }))
    
    model.add(tf.layers.dropout({ rate: 0.2 }))
    
    model.add(tf.layers.dense({
        units: vocabSize,
        activation: 'softmax'
    }))
    
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })
    
    return model
}

// Initialize model only once
function initializeModel() {
    if (!model) {
        const { characterToIndex: cti, indexToCharacter: itc, maxLength: ml, vocabularySize: vs } = preprocessData(trainingData)
        characterToIndex = cti
        indexToCharacter = itc
        maxLength = ml
        vocabularySize = vs
        
        model = createModel(maxLength, vocabularySize)
    }
}

// Training function
async function train() {
    console.log('Training started...')
    const startTime = performance.now()
    
    // Initialize model if it doesn't exist
    initializeModel()
    
    // Prepare data
    const { X, y } = preprocessData(trainingData)
    
    // Train for additional epochs
    console.log(`Continuing training from epoch ${totalEpochsTrained + 1}`)
    
    await model.fit(X, y, {
        initialEpoch: totalEpochsTrained,
        epochs: totalEpochsTrained + AMOUNT_EPOCHS,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`)
            }
        }
    })
    
    // Update total epochs trained
    totalEpochsTrained += AMOUNT_EPOCHS
    
    const endTime = performance.now()
    const trainingTime = (endTime - startTime) / 1000
    console.log(`Training finished! Took ${trainingTime.toFixed(2)} seconds`)
    console.log(`Total epochs trained: ${totalEpochsTrained}`)
}

// Generate new names
function generateName(seed, temperature = 0.5) {
    let currentSequence = seed.toLowerCase()
    
    while (currentSequence.length < 20) {
        const input = Array(maxLength).fill(0);
        [...currentSequence].forEach((char, i) => {
            input[i] = characterToIndex[char] || 0
        })
        
        const inputTensor = tf.tensor2d([input], [1, maxLength])
        const prediction = model.predict(inputTensor)
        const probabilities = prediction.dataSync()
        
        // Apply temperature
        const logProbs = probabilities.map(p => Math.log(p) / temperature)
        const expProbs = logProbs.map(p => Math.exp(p))
        const sum = expProbs.reduce((a, b) => a + b)
        const adjustedProbs = expProbs.map(p => p / sum)
        
        let r = Math.random()
        let idx = 0
        while (r > 0 && idx < vocabularySize) {
            r -= adjustedProbs[idx]
            idx++
        }
        idx--
        
        const nextChar = indexToCharacter[idx]
        if (nextChar === ' ') break
        
        currentSequence += nextChar
        
        inputTensor.dispose()
        prediction.dispose()
    }
    
    return currentSequence
}

function createNames() {
    const newNames = []
    const characters = Object.keys(characterToIndex).join('').replace('-', '')
    
    for (let i = 0; i < 10; i++) {
        const length = Math.floor(Math.random() * 4) + 3
        let seed = ''
        for (let j = 0; j < length; j++) {
            seed += characters[Math.floor(Math.random() * characters.length)]
        }
        
        let newName = generateName(seed)
        if (newName.replace(' ', '').length > 0) {
            newNames.push(newName)
        }
    }
    
    document.querySelector('#names').innerHTML = newNames.join('<br>')
}

// Save model
async function saveModel() {
    try {
        // Save model architecture and weights
        await model.save('downloads://name-generator-model')
        
        // Save additional training metadata
        const metadata = {
            totalEpochsTrained,
            charToIdx: characterToIndex,
            idxToChar: indexToCharacter,
            maxLen: maxLength,
            vocabSize: vocabularySize
        }
        
        const blob = new Blob([JSON.stringify(metadata)], { type: 'application/json' })
        const metadataUrl = URL.createObjectURL(blob)
        const downloadLink = document.createElement('a')
        downloadLink.href = metadataUrl
        downloadLink.download = 'name-generator-metadata.json'
        document.body.appendChild(downloadLink)
        downloadLink.click()
        document.body.removeChild(downloadLink)
        URL.revokeObjectURL(metadataUrl)
        
        console.log('Model and metadata saved successfully!')
    } catch (error) {
        console.error('Error saving model:', error)
        alert('Error saving model file.')
    }
}

// Load model
async function loadModel() {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.multiple = true
    
    input.addEventListener('change', async (e) => {
        try {
            const files = Array.from(e.target.files)
            
            // Find model and metadata files
            const modelFile = files.find(f => f.name.includes('model'))
            const metadataFile = files.find(f => f.name.includes('metadata'))
            
            if (!modelFile || !metadataFile) {
                throw new Error('Please select both model and metadata files')
            }
            
            // Load model
            model = await tf.loadLayersModel(tf.io.browserFiles([modelFile]))
            
            // Recompile the model
            model.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            })
            
            // Load metadata
            const text = await metadataFile.text()
            const metadata = JSON.parse(text)
            
            // Restore metadata
            totalEpochsTrained = metadata.totalEpochsTrained
            characterToIndex = metadata.charToIdx
            indexToCharacter = metadata.idxToChar
            maxLength = metadata.maxLen
            vocabularySize = metadata.vocabSize
            
            console.log('Model and metadata loaded successfully!')
            console.log(`Resumed from epoch ${totalEpochsTrained}`)
        } catch (error) {
            console.error('Error loading files:', error)
            alert(error.message || 'Error loading files.')
        }
    })
    
    input.click()
}

// Event listeners
document.querySelector('#create').addEventListener('click', createNames)
document.querySelector('#train').addEventListener('click', train)
document.querySelector('#load').addEventListener('click', loadModel)
document.querySelector('#save').addEventListener('click', saveModel)