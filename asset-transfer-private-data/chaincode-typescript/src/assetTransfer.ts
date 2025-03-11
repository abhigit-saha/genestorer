/*
 * SPDX-License-Identifier: Apache-2.0
 */
import { Context, Contract, Info, Transaction } from 'fabric-contract-api';
import stringify from 'json-stringify-deterministic';
import sortKeysRecursive from 'sort-keys-recursive';
import { Asset } from './asset';
import { AssetPrivateDetails } from './assetTransferDetails';
import {
    TransientAssetDelete,
    TransientAssetProperties,
} from './assetTransferTransientInput';

const assetCollection = 'assetCollection';

@Info({
    title: 'AssetTransfer',
    description: 'Smart contract for trading assets',
})
export class AssetTransfer extends Contract {
    // CreateAsset issues a new asset to the world state with given details.
    @Transaction()
    public async CreateAsset(ctx: Context): Promise<void> {
        const transientMap = ctx.stub.getTransient();
        const assetProperties = new TransientAssetProperties(transientMap);

        // Check if asset already exists

        // Get ID of submitting client identity
        // const clientID = ctx.clientIdentity.getID();

        // Verify that the client is submitting request to peer in their organization
        // This is to ensure that a client from another org doesn't attempt to read or
        // write private data from this peer.
        this.verifyClientOrgMatchesPeerOrg(ctx);

        // Save asset details to collection visible to owning organization
        const assetPrivateDetails: AssetPrivateDetails = {
            ID: assetProperties.assetID,
            data: assetProperties.data,
        };
        // Get collection name for this organization.
        const orgCollection = this.getCollectionName(ctx);
        // Put asset appraised value into owners org specific private data collection
        console.log(
            'Put: collection %v, ID %v',
            orgCollection,
            assetProperties.assetID
        );
        await ctx.stub.putPrivateData(
            orgCollection,
            assetPrivateDetails.ID,
            Buffer.from(stringify(sortKeysRecursive(assetPrivateDetails)))
        );
    }

    @Transaction()
    // DeleteAsset can be used by the owner of the asset to delete the asset
    public async DeleteAsset(ctx: Context): Promise<void> {
        // Value is private, therefore it gets passed in transient field
        const transientMap = ctx.stub.getTransient();
        const assetDelete = new TransientAssetDelete(transientMap);

        // Verify that the client is submitting request to peer in their organization
        this.verifyClientOrgMatchesPeerOrg(ctx);

        console.log('Deleting Asset: ' + assetDelete.assetID);
        // get the asset from chaincode state
        const orgCollection = this.getCollectionName(ctx);

        const valAsbytes = await ctx.stub.getPrivateData(
            orgCollection,
            assetDelete.assetID
        );
        if (valAsbytes.length === 0) {
            throw new Error('asset not found: ' + assetDelete.assetID);
        }

        await ctx.stub.deletePrivateData(orgCollection, assetDelete.assetID);
    }

    /*
        GETTERS
    */

    // ReadAsset reads the information from collection
    @Transaction()
    public async ReadAsset(ctx: Context, id: string): Promise<Asset> {
        const orgCollection = this.getCollectionName(ctx);

        // Check if asset already exists

        const assetAsBytes = await ctx.stub.getPrivateData(orgCollection, id);
        // No Asset found, return empty response
        if (assetAsBytes.length === 0) {
            throw new Error(
                id + ' does not exist in collection ' + assetCollection
            );
        }
        return Asset.fromBytes(assetAsBytes);
    }

    // ReadAssetPrivateDetails reads the asset private details in organization specific collection

    // GetAssetByRange performs a range query based on the start and end keys provided. Range
    // queries can be used to read data from private data collections, but can not be used in
    // a transaction that also writes to private data.

    /*
        HELPERS
    */
    // verifyAgreement is an internal helper function used by TransferAsset to verify
    // that the transfer is being initiated by the owner and that the buyer has agreed
    // to the same appraisal value as the owner
    public async verifyAgreement(
        ctx: Context,
        assetID: string,
        owner: string,
        buyerMSP: string
    ): Promise<void> {
        // Check 1: verify that the transfer is being initiatied by the owner
        // Get ID of submitting client identity
        const clientID = ctx.clientIdentity.getID();
        if (clientID !== owner) {
            throw new Error(
                `error: submitting client(${clientID}) identity does not own asset ${assetID}.Owner is ${owner}`
            );
        }
        // Check 2: verify that the buyer has agreed to the appraised value
        // Get collection names
        const collectionOwner = this.getCollectionName(ctx); // get owner collection from caller identity

        const collectionBuyer = buyerMSP + 'PrivateCollection'; // get buyers collection
        // Get hash of owners agreed to value
        const ownerAppraisedValueHash = await ctx.stub.getPrivateDataHash(
            collectionOwner,
            assetID
        );

        if (ownerAppraisedValueHash.length === 0) {
            throw new Error(
                `hash of appraised value for ${assetID} does not exist in collection ${collectionOwner}`
            );
        }
        // Get hash of buyers agreed to value
        const buyerAppraisedValueHash = await ctx.stub.getPrivateDataHash(
            collectionBuyer,
            assetID
        );
        if (buyerAppraisedValueHash.length === 0) {
            throw new Error(
                `hash of appraised value for ${assetID} does not exist in collection ${collectionBuyer}. AgreeToTransfer must be called by the buyer first`
            );
        }
        // Verify that the two hashes match
        if (
            ownerAppraisedValueHash.toString() !==
            buyerAppraisedValueHash.toString()
        ) {
            throw new Error(
                `hash for appraised value for owner ${Buffer.from(
                    ownerAppraisedValueHash
                ).toString(
                    'hex'
                )} does not match value for seller ${Buffer.from(
                    buyerAppraisedValueHash
                ).toString('hex')}`
            );
        }
    }
    // getCollectionName is an internal helper function to get collection of submitting client identity.
    public getCollectionName(ctx: Context): string {
        // Get the MSP ID of submitting client identity
        const clientMSPID = ctx.clientIdentity.getMSPID();
        // Create the collection name
        const orgCollection = clientMSPID + 'PrivateCollection';

        return orgCollection;
    }
    // Get ID of submitting client identity
    public submittingClientIdentity(ctx: Context): string {
        const b64ID = ctx.clientIdentity.getID();

        // base64.StdEncoding.DecodeString(b64ID);
        const decodeID = Buffer.from(b64ID, 'base64').toString('binary');

        return String(decodeID);
    }
    // verifyClientOrgMatchesPeerOrg is an internal function used verify client org id and matches peer org id.
    public verifyClientOrgMatchesPeerOrg(ctx: Context): void {
        const clientMSPID = ctx.clientIdentity.getMSPID();

        const peerMSPID = ctx.stub.getMspID();

        if (clientMSPID !== peerMSPID) {
            throw new Error(
                'client from org %v is not authorized to read or write private data from an org ' +
                    clientMSPID +
                    ' peer ' +
                    peerMSPID
            );
        }
    }
    // =======Rich queries =========================================================================
    // Two examples of rich queries are provided below (parameterized query and ad hoc query).
    // Rich queries pass a query string to the state database.
    // Rich queries are only supported by state database implementations
    //  that support rich query (e.g. CouchDB).
    // The query string is in the syntax of the underlying state database.
    // With rich queries there is no guarantee that the result set hasn't changed between
    //  endorsement time and commit time, aka 'phantom reads'.
    // Therefore, rich queries should not be used in update transactions, unless the
    // application handles the possibility of result set changes between endorsement and commit time.
    // Rich queries can be used for point-in-time queries against a peer.
    // ============================================================================================

    // ===== Example: Parameterized rich query =================================================

    // QueryAssetByOwner queries for assets based on assetType, owner.
    // This is an example of a parameterized query where the query logic is baked into the chaincode,
    // and accepting a single query parameter (owner).
    // Only available on state databases that support rich query (e.g. CouchDB)
    // =========================================================================================
    public async QueryAssetByOwner(
        ctx: Context,
        assetType: string,
        owner: string
    ): Promise<Asset[]> {
        const queryString = `{'selector':{'objectType':'${assetType}','owner':'${owner}'}}`;

        return await this.getQueryResultForQueryString(ctx, queryString);
    }

    public QueryAssets(ctx: Context, queryString: string): Promise<Asset[]> {
        return this.getQueryResultForQueryString(ctx, queryString);
    }

    public async getQueryResultForQueryString(
        ctx: Context,
        queryString: string
    ): Promise<Asset[]> {
        const resultsIterator = ctx.stub.getPrivateDataQueryResult(
            assetCollection,
            queryString
        );

        const results: Asset[] = [];

        for await (const res of resultsIterator) {
            const asset = Asset.fromBytes(res.value);
            results.push(asset);
        }

        return results;
    }
}
